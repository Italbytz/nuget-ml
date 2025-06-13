using System.Globalization;
using System.Reflection;
using System.Text;
using Italbytz.ML.Data;
using Italbytz.ML.ModelBuilder.Configuration;
using Microsoft.ML;

namespace Italbytz.ML;

public class Explainer(
    ITransformer model,
    IDataView data,
    ScenarioType scenario,
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
        if (scenario != ScenarioType.Classification)
            throw new InvalidOperationException(
                "Permutation feature importance is currently only supported for classification scenarios.");
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var scoredDataView = model.Transform(data);
        return
            mlContext.MulticlassClassification
                .GetPermutationFeatureImportanceTable(model,
                    scoredDataView, labelColumnName, metric);
    }

    public string GetCeterisParibusTable<ModelInput, ModelOutput>(
        int featureIndex = 0, int gridCells = 100)
        where ModelInput : class, new()
        where ModelOutput : class, new()
    {
        if (scenario != ScenarioType.Classification)
            throw new InvalidOperationException(
                "Ceteris Paribus is currently only supported for classification scenarios.");
        if (featureIndex < 0)
            throw new ArgumentOutOfRangeException(nameof(featureIndex),
                "Feature index must be non-negative.");

        var sb = new StringBuilder();
        var mlContext = new MLContext();

        var predictionEngine =
            mlContext.Model
                .CreatePredictionEngine<ModelInput,
                    ModelOutput>(
                    model);

        var features = data.GetFeatures<ModelInput>(labelColumnName);
        if (features == null || features.Count == 0)
            throw new InvalidOperationException(
                "No features found in the provided data. " +
                "Ensure that the data contains valid features for Ceteris Paribus analysis.");

        var selectedFeature =
            features.FirstOrDefault(f => f.ColumnIndex == featureIndex);
        if (selectedFeature == null)
            throw new ArgumentException(
                $"No feature found with ColumnIndex {featureIndex}.",
                nameof(featureIndex));

        if (selectedFeature is not NumericalFeature numericalFeature)
            throw new ArgumentException(
                "Ceteris Paribus is currently only supported for numerical features.",
                nameof(featureIndex));

        // Generate grid values for the selected feature
        var minValue = numericalFeature.ValueRange[0];
        var maxValue = numericalFeature.ValueRange[1];
        var step = (maxValue - minValue) / (gridCells - 1);
        var gridValues = new float[gridCells];
        for (var i = 0; i < gridCells; i++) gridValues[i] = minValue + i * step;


        var dataArray =
            mlContext.Data.CreateEnumerable<ModelInput>(data, true).ToArray();

        // Create a header for the CSV output
        sb.AppendLine("Feature,Score,Class");

        // Iterate over each row
        var scoreCount = 0;
        foreach (var gridValue in gridValues)
        {
            var scores = new List<float[]>();

            foreach (var row in dataArray)
            {
                // Create a copy of the row to modify
                var modifiedRow = new ModelInput();
                // Copy properties from the original row to the modified row
                foreach (var prop in typeof(ModelInput).GetProperties(
                             BindingFlags.Public | BindingFlags.Instance))
                    if (prop.CanWrite)
                        prop.SetValue(modifiedRow,
                            prop.Name == selectedFeature.PropertyName
                                ? gridValue
                                : prop.GetValue(row));
                var prediction = predictionEngine.Predict(modifiedRow);
                var scoreProperty = prediction.GetType().GetProperty("Score");
                var scoreArray = scoreProperty?.GetValue(prediction) as float[];
                if (scoreArray == null)
                    throw new InvalidOperationException(
                        "Prediction output does not contain a 'Score' property.");
                if (scoreCount == 0) scoreCount = scoreArray.Length;

                scores.Add(scoreArray);
            }

            for (var i = 0; i < scoreCount; i++)
            {
                var averageScore = scores
                    .Select(s => s[i])
                    .Average();
                sb.AppendLine(
                    $"{gridValue.ToString(CultureInfo.InvariantCulture)},{averageScore.ToString(CultureInfo.InvariantCulture)},Class{i}");
            }
        }

        return sb.ToString();
    }

    private static IEnumerable<(string name, object value)>
        GetPropertyValues<T>(T obj, BindingFlags flags)
    {
        return from p in typeof(T).GetProperties(flags)
            where p.GetIndexParameters().Length == 0 //To filter out indexers
            select (p.Name, p.GetValue(obj, null));
    }
}