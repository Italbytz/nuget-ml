using System.Globalization;
using System.Text;
using Italbytz.ML.Data;
using Microsoft.ML;

namespace Italbytz.ML;

/// <summary>
///     Extension methods for <see cref="MulticlassClassificationCatalog" /> to
///     provide additional functionality for multiclass classification models.
/// </summary>
public static class MulticlassClassificationCatalogExtensions
{
    /// <summary>
    ///     Generates a comma-separated values (CSV) formatted string containing
    ///     permutation feature importance information for a multiclass classification
    ///     model.
    /// </summary>
    /// <param name="model">The trained model to evaluate.</param>
    /// <param name="data">The data to use for calculating feature importance.</param>
    /// <param name="labelColumnName">The name of the label column.</param>
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
    public static string GetPermutationFeatureImportanceTable(
        this MulticlassClassificationCatalog catalog,
        ITransformer model, IDataView data, string? labelColumnName,
        Metric metric)
    {
        var sb = new StringBuilder();
        sb.AppendLine(
            "Feature, Importance");
        var permutationFeatureImportance =
            catalog
                .PermutationFeatureImportance(
                    model,
                    data,
                    labelColumnName);
        foreach (var (key, value) in permutationFeatureImportance)
        {
            var metricValue = metric switch
            {
                Metric.MacroAccuracy => value.MacroAccuracy.Mean,
                Metric.MicroAccuracy => value.MicroAccuracy.Mean,
                Metric.LogLoss => value.LogLoss.Mean,
                _ => 0.0
            };
            if (metricValue == 0.0f)
                continue;
            var importance = metricValue * -1;
            var valueString = importance.ToString(
                CultureInfo.InvariantCulture);
            sb.AppendLine(
                $"{key}, {valueString}");
        }

        return sb.ToString();
    }
}