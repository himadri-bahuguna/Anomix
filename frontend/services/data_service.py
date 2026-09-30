from typing import Any

from data import mock_data


class DataService:
    """Stable UI-facing facade; replace these methods with API adapters later."""

    def get_overview(self) -> dict[str, Any]:
        return mock_data.get_overview_data()

    def get_component_list(self) -> list[str]:
        return mock_data.get_component_list()

    def get_component_data(self, component_id: str) -> dict[str, Any]:
        return mock_data.get_component_data(component_id)

    def get_module_a_results(self) -> dict[str, Any]:
        return mock_data.get_module_a_results()

    def get_module_b_results(self) -> dict[str, Any]:
        return mock_data.get_module_b_results()

    def get_ml_metrics(self) -> dict[str, Any]:
        return mock_data.get_ml_metrics()

    def get_dataset_status(self) -> dict[str, Any]:
        return mock_data.get_dataset_status()
