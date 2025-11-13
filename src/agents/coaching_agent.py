# mlops_sport_analytics/src/agents/coaching_agent.py
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.chat_completion_client_base import (
    ChatCompletionClientBase,
)
from semantic_kernel.contents import ChatMessageContent, AuthorRole
from semantic_kernel.functions import kernel_function, KernelFunctionFromPrompt
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.connectors.ai.ollama import OllamaPromptExecutionSettings
from semantic_kernel.functions import KernelArguments
import yaml
import os
from typing import List, Dict
import asyncio
import aiohttp
import requests


def load_config(config_path: str = None) -> Dict:
    """Charge config (comme Étape 2)."""
    if config_path is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        config_path = os.path.join(base_dir, "config", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# Custom OllamaChatCompletion (simplified version)
class OllamaChatCompletion(ChatCompletionClientBase):
    """
    Connecteur personnalisé pour Ollama.
    """

    def __init__(
        self,
        service_id: str,
        ai_model_id: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
    ):
        super().__init__(service_id=service_id, ai_model_id=ai_model_id)
        object.__setattr__(self, "base_url", base_url)
        object.__setattr__(self, "model", ai_model_id)

    async def get_chat_message_contents(
        self, chat_history, settings, **kwargs
    ) -> list[ChatMessageContent]:
        """Récupère la réponse du modèle Ollama."""
        messages = [{"role": m.role.value, "content": m.content} for m in chat_history]
        # CORRECTION : Utiliser getattr() pour accéder aux attributs de l'objet PromptExecutionSettings
        temperature = getattr(settings, "temperature", 0.3) if settings else 0.3
        max_tokens = (
            getattr(settings, "max_tokens", 800) if settings else 800
        )  # Tokens ↑ pour rapport complet
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "stream": False,
                    "options": {"temperature": temperature, "num_predict": max_tokens},
                },
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return [
                        ChatMessageContent(
                            role=AuthorRole.ASSISTANT,
                            content=result["message"]["content"],
                        )
                    ]
                else:
                    raise Exception(f"Erreur Ollama : {await response.text()}")


class MLInsightsSkill:
    """Plugin pour insights ML."""

    def __init__(self, model, scaler):
        self.model = model
        self.scaler = scaler

    @kernel_function(
        description="Prédit impacts pour un match ID", name="predict_impact"
    )
    def predict_impact(self, match_id: str) -> str:
        """Skill ML : Filtre CSV, prédit impacts."""
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
        csv_path = os.path.join(base_dir, "data", "processed", "all_matches_merged.csv")
        print(f"Debug: CSV Path = {csv_path}")
        try:
            df = pd.read_csv(csv_path)
            print(
                f"Debug: CSV loaded, shape={df.shape}, match_ids sample={df['match_id'].unique()[:3]}"
            )
            match_df = df[df["match_id"] == match_id].copy()
            players_df = match_df[
                match_df["player_impact"].notna()
                & ~match_df["name"].str.contains("tot_s", na=False)
            ]
            if players_df.empty:
                return f"Aucun data pour match {match_id}. Vérifiez l'ID dans le CSV."
            # FIX: Drop duplicates par name pour éviter reps (multi-runs)
            players_df = players_df.drop_duplicates(subset=["name"])
            # Features...
            num_cols = players_df.select_dtypes(include=[np.number]).columns
            feature_cols = [
                col
                for col in num_cols
                if col.startswith(("s", "tot_s")) and col not in ["player_impact"]
            ]
            feature_cols += ["off_efficiency", "possessions"]
            available_features = [
                col for col in feature_cols if col in players_df.columns
            ]
            X = players_df[available_features].fillna(0)
            X_scaled = self.scaler.transform(X)
            impacts_pred = self.model.predict(X_scaled)
            players_df["impact_pred"] = impacts_pred
            # FIX: Top3 unique, sorted
            top3_df = players_df.nlargest(3, "impact_pred")[
                ["name", "impact_pred", "sPoints", "sReboundsTotal"]
            ]
            top3 = top3_df.to_dict("records")
            key_stats_ml = f"Turnovers avg: {players_df['sTurnovers'].mean():.1f}, Off Eff: {players_df['off_efficiency'].mean():.2f}"
            return f"Top3: {top3} | Key Stats: {key_stats_ml}"
        except FileNotFoundError:
            return f"CSV non trouvé: {csv_path}."
        except Exception as e:
            return f"Erreur prédiction: {str(e)}"


class CoachingAgent:
    def __init__(self, run_id: str = "1c4e67cdc060408582478db7b57da2c7"):
        config = load_config()
        self.tracking_uri = config["mlflow_tracking_uri"]
        mlflow.set_tracking_uri(self.tracking_uri)
        self.run_id = run_id
        self.experiment_name = "sport_impact_v1"
        # Load ML models
        try:
            self.model = mlflow.sklearn.load_model(
                f"runs:/{self.run_id}/random_forest_model"
            )
            self.scaler = mlflow.sklearn.load_model(f"runs:/{self.run_id}/scaler")
        except Exception as e:
            raise ValueError(f"Failed to load MLflow models: {e}")
        # Semantic Kernel setup
        self.kernel = Kernel()
        if config.get("openai_key") and config["openai_key"].strip():
            from semantic_kernel.connectors.ai.open_ai import OpenAIChatCompletion

            self.chat_service = OpenAIChatCompletion(
                service_id="openai_service",
                ai_model_id="gpt-3.5-turbo",
                api_key=config["openai_key"],
            )
            print("✅ Using OpenAI")
        else:
            ollama_model = config.get("ollama_model", "llama3.1:8b").strip()
            ollama_host = config.get("ollama_host", "http://localhost:11434")
            self.chat_service = OllamaChatCompletion(
                service_id="ollama_service",
                ai_model_id=ollama_model,
                base_url=ollama_host,
            )
            # Health check
            try:
                response = requests.get(f"{ollama_host}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [model.get("name", "") for model in models]
                    if any(ollama_model in name for name in model_names):
                        print(f"✅ Ollama ready with {ollama_model}")
                    else:
                        print(f"⚠️ Model {ollama_model} not found in Ollama")
                else:
                    print("⚠️ Could not connect to Ollama API")
            except Exception as e:
                print(f"⚠️ Could not connect to Ollama: {e}")
        # Ajout du service au kernel
        self.kernel.add_service(self.chat_service)
        # Register skills
        self.register_ml_skill()
        self.register_llm_skill()

    def register_ml_skill(self):
        """Skill 1: Prédit impacts via ML."""
        self.ml_plugin = MLInsightsSkill(self.model, self.scaler)
        self.kernel.add_plugin(self.ml_plugin, plugin_name="ml_insights")

    def register_llm_skill(self):
        """Skill 2: Génère rapport avec MCP."""
        template = """Tu es un coach IA expert en basketball. Base-toi UNIQUEMENT sur les data fournies (insights ML + stats match). SI AUCUNE DATA CONCRÈTE (erreur ou vide), dis-le explicitement et limite-toi à des conseils généraux courts – PAS D'INVENTIONS DE JOUEURS/STATS.
        Génère un rapport post-mortem concis et actionable :
        1. Top 3 joueurs par impact prédit (avec raison : ex. 'via points/rebonds'). Si pas de data, note-le.
        2. Analyse collective (ex. : turnovers élevés → perte possession). Si pas de data, note-le.
        3. Recommandations entraînement (2-3 spécifiques, mesurables).
        Sois grounded : Pas d'inventions. Actionnable : Mesures concrètes.
        Match ID: {{$match_id}}
        Insights ML: {{$ml_insights}}
        Stats clés: {{$key_stats}}
        Rapport en français."""
        # ✅ IMPROVED: Prompt plus strict pour éviter hallucinations si ML erreur
        prompt_config = PromptTemplateConfig(
            template=template,
            description="Génère un rapport d'analyse d'un match de basketball.",
            name="generate_report",
        )
        self.report_function = KernelFunctionFromPrompt(
            function_name="generate_report", prompt_template_config=prompt_config
        )
        # CORRECTION : Lier explicitement la fonction au service pour éviter l'erreur "No service found"
        self.kernel.add_function(
            plugin_name="report_gen",
            function=self.report_function,
            target_service=self.chat_service,
        )

    async def generate_report(self, match_id: str) -> str:
        """Chaîne skills : ML → LLM pour rapport complet."""
        # Skill 1: ML insights
        ml_insights_result = await self.kernel.invoke(
            plugin_name="ml_insights", function_name="predict_impact", match_id=match_id
        )
        ml_insights = str(ml_insights_result)
        print(f"Debug: ML Insights = {ml_insights}")  # Debug print
        # AMÉLIORATION : Extraction des Top3 (ml_insights_data) et des Key Stats (key_stats_data)
        if " | Key Stats: " in ml_insights:
            parts = ml_insights.split(" | Key Stats: ")
            ml_insights_data = parts[0].replace("Top3: ", "")
            key_stats_data = parts[1]
        else:
            ml_insights_data = ml_insights
            key_stats_data = "Statistiques collectives non disponibles (Turnovers/Off. Eff. non calculés ou erreur)."  # Fallback
        print(
            f"Debug: LLM Vars - Insights: {ml_insights_data[:100]}... | Stats: {key_stats_data}"
        )  # ✅ IMPROVED: Debug LLM inputs
        # Skill 2: LLM report
        variables = KernelArguments()
        variables["match_id"] = match_id
        variables["ml_insights"] = ml_insights_data  # Insights joueurs/Top3
        variables["key_stats"] = key_stats_data  # Stats collectives
        # Le 'settings' n'est plus nécessaire ici car target_service est défini dans register_llm_skill.
        report_result = await self.kernel.invoke(
            plugin_name="report_gen",
            function_name="generate_report",
            arguments=variables,
        )
        return str(report_result)

    def generate_report_sync(self, season):
        """Sync wrapper for async report generation"""
        import asyncio

        return asyncio.run(self.generate_report(season))


# Test main
async def main():
    try:
        # Assurez-vous d'avoir un 'run_id' valide et les modèles MLflow stockés
        agent = CoachingAgent()
        print("🤖 Agent initialisé, génération du rapport...")
        # ✅ FIXED: Utilise un match_id valide du sample (change si besoin pour d'autres)
        report = await agent.generate_report("2021")  # Premier du sample: '2021'
        print("=" * 50)
        print("📊 RAPPORT GÉNÉRÉ :")
        print("=" * 50)
        print(report)
    except Exception as e:
        print(f"❌ Erreur: {e}")


if __name__ == "__main__":
    asyncio.run(main())
