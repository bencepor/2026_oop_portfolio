from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import pandas as pd
import plotly.express as px
import plotly.io as pio
from shiny import App, reactive, render, ui


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration container for paths and mood classes."""

    data_dir: Path
    moods: List[str]


class CsvRepository:
    """Single responsibility repository for reading local CSV assets."""

    def __init__(self, data_dir: Path):
        self._data_dir = data_dir

    def load(self, filename: str) -> pd.DataFrame:
        path = self._data_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing required data file: {path}")
        return pd.read_csv(path)


@dataclass(frozen=True)
class ModelResult:
    model: str
    feature_set: str
    accuracy: float
    precision_macro: float
    recall_macro: float
    f1_macro: float
    samples_per_class: int
    notes: str


class ExperimentService:
    """Application service that maps tabular data into domain objects."""

    def __init__(self, repository: CsvRepository):
        self._repository = repository
        self._model_results_df = repository.load("model_results.csv")
        self._mood_profiles_df = repository.load("mood_feature_profiles.csv")
        self._confusion_df = repository.load("confusion_matrix_rf.csv")

    @property
    def model_results_df(self) -> pd.DataFrame:
        return self._model_results_df.copy()

    @property
    def mood_profiles_df(self) -> pd.DataFrame:
        return self._mood_profiles_df.copy()

    @property
    def confusion_df(self) -> pd.DataFrame:
        return self._confusion_df.copy()

    def available_models(self) -> List[str]:
        return self._model_results_df["model"].tolist()

    def available_metrics(self) -> List[str]:
        return ["accuracy", "precision_macro", "recall_macro", "f1_macro"]

    def get_model_result(self, model_name: str) -> ModelResult:
        row = self._model_results_df[self._model_results_df["model"] == model_name]
        if row.empty:
            raise ValueError(f"Unknown model: {model_name}")
        record = row.iloc[0].to_dict()
        return ModelResult(**record)


class PlotBuilder(ABC):
    """Abstract chart strategy to keep plotting extensible."""

    @abstractmethod
    def build_html(self) -> str:
        pass


class ModelComparisonPlotBuilder(PlotBuilder):
    def __init__(self, model_df: pd.DataFrame, metric: str):
        self._model_df = model_df
        self._metric = metric

    def build_html(self) -> str:
        fig = px.bar(
            self._model_df,
            x="model",
            y=self._metric,
            color="model",
            text=self._metric,
            title=f"Model comparison by {self._metric.replace('_', ' ').title()}",
            range_y=[0, 1],
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%")
        return pio.to_html(fig, include_plotlyjs="cdn", full_html=False)


class MoodProfilePlotBuilder(PlotBuilder):
    def __init__(self, mood_df: pd.DataFrame, mood: str):
        self._mood_df = mood_df
        self._mood = mood

    def build_html(self) -> str:
        selected = self._mood_df[self._mood_df["mood"] == self._mood].copy()
        features = ["mfcc_energy", "chroma_warmth", "spectral_contrast", "zero_crossing_rate"]
        long_df = selected.melt(
            id_vars=["mood"],
            value_vars=features,
            var_name="feature",
            value_name="value",
        )
        fig = px.line_polar(
            long_df,
            r="value",
            theta="feature",
            line_close=True,
            title=f"Feature profile for '{self._mood}' mood",
            range_r=[0, 1],
        )
        fig.update_traces(fill="toself")
        return pio.to_html(fig, include_plotlyjs=False, full_html=False)


class ConfusionMatrixPlotBuilder(PlotBuilder):
    def __init__(self, confusion_df: pd.DataFrame):
        self._confusion_df = confusion_df

    def build_html(self) -> str:
        matrix = self._confusion_df.pivot(
            index="true_mood",
            columns="pred_mood",
            values="count",
        )
        fig = px.imshow(
            matrix,
            text_auto=True,
            color_continuous_scale="Blues",
            title="Random Forest confusion matrix (illustrative)",
        )
        fig.update_layout(xaxis_title="Predicted mood", yaxis_title="True mood")
        return pio.to_html(fig, include_plotlyjs=False, full_html=False)


class PortfolioUIFactory:
    def __init__(self, service: ExperimentService, config: ExperimentConfig):
        self._service = service
        self._config = config

    def build(self):
        return ui.page_fluid(
            ui.h2("Music Mood Classification Portfolio App"),
            ui.p(
                "This Shiny app summarizes the workflow from classification.ipynb using",
                " object-oriented Python classes and reusable components.",
            ),
            ui.layout_sidebar(
                ui.sidebar(
                    ui.input_select(
                        "model",
                        "Model:",
                        choices=self._service.available_models(),
                    ),
                    ui.input_select(
                        "metric",
                        "Comparison metric:",
                        choices=self._service.available_metrics(),
                        selected="accuracy",
                    ),
                    ui.input_select(
                        "mood",
                        "Mood profile:",
                        choices=self._config.moods,
                        selected=self._config.moods[0],
                    ),
                ),
                ui.value_box(title="Accuracy", value=ui.output_text("accuracy_text")),
                ui.value_box(title="Macro F1", value=ui.output_text("f1_text")),
                ui.output_ui("model_notes"),
                ui.output_ui("comparison_plot"),
                ui.output_ui("mood_plot"),
                ui.output_ui("confusion_plot"),
                ui.output_data_frame("results_table"),
            ),
        )


class PortfolioServerController:
    def __init__(self, service: ExperimentService):
        self._service = service

    def register(self, input, output, session):
        @reactive.Calc
        def selected_result() -> ModelResult:
            return self._service.get_model_result(input.model())

        @output
        @render.text
        def accuracy_text():
            return f"{selected_result().accuracy:.0%}"

        @output
        @render.text
        def f1_text():
            return f"{selected_result().f1_macro:.0%}"

        @output
        @render.ui
        def model_notes():
            result = selected_result()
            return ui.tags.div(
                ui.h4("Selected model details"),
                ui.tags.ul(
                    ui.tags.li(f"Feature set: {result.feature_set}"),
                    ui.tags.li(f"Samples per mood: {result.samples_per_class}"),
                    ui.tags.li(f"Notes: {result.notes}"),
                ),
            )

        @output
        @render.ui
        def comparison_plot():
            builder = ModelComparisonPlotBuilder(
                self._service.model_results_df,
                input.metric(),
            )
            return ui.HTML(builder.build_html())

        @output
        @render.ui
        def mood_plot():
            builder = MoodProfilePlotBuilder(self._service.mood_profiles_df, input.mood())
            return ui.HTML(builder.build_html())

        @output
        @render.ui
        def confusion_plot():
            builder = ConfusionMatrixPlotBuilder(self._service.confusion_df)
            return ui.HTML(builder.build_html())

        @output
        @render.data_frame
        def results_table():
            table_df = self._service.model_results_df.copy()
            display_columns: Dict[str, str] = {
                "model": "Model",
                "feature_set": "Feature Set",
                "accuracy": "Accuracy",
                "precision_macro": "Precision (Macro)",
                "recall_macro": "Recall (Macro)",
                "f1_macro": "F1 (Macro)",
            }
            return render.DataGrid(
                table_df[list(display_columns.keys())].rename(columns=display_columns),
                filters=True,
            )


class PortfolioApp:
    def __init__(self):
        data_dir = Path(__file__).parent / "data"
        config = ExperimentConfig(
            data_dir=data_dir,
            moods=["aggressive", "dramatic", "happy", "romantic", "sad"],
        )
        repository = CsvRepository(config.data_dir)
        service = ExperimentService(repository)

        self._ui = PortfolioUIFactory(service, config)
        self._server = PortfolioServerController(service)

    def build_ui(self):
        return self._ui.build()

    def build_server(self, input, output, session):
        self._server.register(input, output, session)


portfolio_app = PortfolioApp()
app = App(portfolio_app.build_ui(), portfolio_app.build_server)
