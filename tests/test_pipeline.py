# mlops_sport_analytics/tests/test_pipeline.py
#!/usr/bin/env python3
"""
Tests d'intégration pour le pipeline complet
"""
import pytest
import asyncio
import sys
import os
from unittest.mock import Mock, patch, AsyncMock
from airflow.models import DagBag

# Ajoute le chemin source
# Note: Cette ligne est essentielle pour que les imports des modules src.* fonctionnent.
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))


class TestPipeline:
    """Tests pour le pipeline MLOps complet"""

    def test_data_processing(self):
        """Test du traitement des données (vérification de la configuration)"""
        from src.data.process_data import load_config

        config = load_config()
        assert "data_dir" in config
        assert "mlflow_tracking_uri" in config

    @patch("src.models.train_model.mlflow.start_run")
    @patch("src.models.train_model.mlflow.log_params")
    @patch("src.models.train_model.mlflow.log_metric")
    @patch("src.models.train_model.mlflow.sklearn.log_model")
    def test_model_training(self, *mocks):
        """Test de l'entraînement du modèle"""
        from src.models.train_model import train_and_log_model

        results = train_and_log_model(run_name="test_run")
        assert "test_r2" in results
        assert results["test_r2"] >= 0
        assert "run_id" in results

    @pytest.mark.asyncio
    @patch("src.agents.coaching_agent.Kernel")
    @patch("src.agents.coaching_agent.OllamaChatCompletion")
    # CORRECTION MLFLOW : Mocker l'appel de chargement des modèles
    @patch("src.agents.coaching_agent.mlflow.sklearn.load_model", return_value=Mock())
    async def test_agent_report(self, mock_load_model, mock_ollama, mock_kernel):
        """Test de la génération de rapport"""

        # 1. Définir le résultat du ML Skill (Simule la chaîne de sortie du MLInsightSkill)
        ml_skill_str_result = Mock()
        ml_skill_str_result.__str__ = lambda self=ml_skill_str_result: (
            'Top3: [{"name": "PlayerA", "impact_pred": 0.9}] | Key Stats: Turnovers avg: 5.0, Off Eff: 1.1'
        )

        # 2. Définir le résultat du LLM Skill (Simule le rapport final)
        llm_skill_str_result = Mock()
        llm_skill_str_result.__str__ = (
            lambda self=llm_skill_str_result: "Rapport Mocké Final"
        )

        # CORRECTION ASYNCHRONE CLÉ :
        # Configurer la méthode .invoke du mock Kernel comme un AsyncMock avec la séquence des résultats.
        # Ceci résout le "TypeError: object AsyncMock can't be used in 'await' expression"
        mock_kernel.return_value.invoke = AsyncMock(
            side_effect=[
                ml_skill_str_result,  # Premier appel: ML Skill
                llm_skill_str_result,  # Second appel: LLM Skill
            ]
        )

        from src.agents.coaching_agent import CoachingAgent

        # L'instanciation réussit
        agent = CoachingAgent(run_id="test_run_123")

        # L'appel asynchrone fonctionne maintenant
        report = await agent.generate_report("2021")

        assert "Rapport Mocké Final" in report

        # Vérification optionnelle
        assert mock_kernel.return_value.invoke.call_count == 2

    def test_pipeline_integration(self):
        """Test d'intégration du pipeline complet (vérification des imports)"""
        from src.data import process_data
        from src.models import train_model
        from src.agents import coaching_agent

        assert process_data is not None
        assert train_model is not None
        assert coaching_agent is not None


def test_dag_loading():
    """Test que le DAG Airflow se charge correctement"""

    # Construction du chemin absolu pour la robustesse
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dag_folder = os.path.join(repo_root, "airflow", "dags")

    dag_bag = DagBag(dag_folder=dag_folder, include_examples=False)

    # Vérifiez l'absence d'erreurs d'importation
    assert (
        len(dag_bag.import_errors) == 0
    ), f"Erreurs d'importation dans le DAG: {dag_bag.import_errors}"

    # Vérification de l'ID du DAG
    assert "sport_analytics_pipeline" in dag_bag.dag_ids


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
