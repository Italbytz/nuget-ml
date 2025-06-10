using Italbytz.ML.Data;
using Italbytz.ML.ModelBuilder.Configuration;
using Microsoft.ML;

namespace Italbytz.ML;

public class Explainer(
    ITransformer mlModel,
    IDataView dataView,
    ScenarioType scenarioType,
    string labelColumnName = DefaultColumnNames.Label)
{
    /// <summary>
    ///     Generates a comma-separated values (CSV) formatted string containing
    ///     permutation feature importance information.
    /// </summary>
    /// <param name="metric">
    ///     The evaluation metric to use for determining feature
    ///     importance.
    /// </param>
    /// <returns>A CSV formatted string with feature names and their importance values.</returns>
    /// <remarks>
    ///     The table contains two columns: Feature and Importance.
    ///     Importance values are negated to show positive values for important
    ///     features.
    ///     Features with zero importance are excluded from the results.
    /// </remarks>
    public string GetPermutationFeatureImportanceTable(Metric metric)
    {
        if (scenarioType != ScenarioType.Classification)
            throw new InvalidOperationException(
                "Permutation feature importance is currently only supported for classification scenarios.");
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        return
            mlContext.MulticlassClassification
                .GetPermutationFeatureImportanceTable(mlModel,
                    dataView, labelColumnName, metric);
    }
}