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

    public string GetCeterisParibusScript<ModelInput, ModelOutput>(
        int featureIndex = 0) where ModelInput : class, new()
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

        var dataEnumerable =
            mlContext.Data.CreateEnumerable<ModelInput>(data, true);

        // Iterate over each row
        foreach (var row in dataEnumerable)
        {
            var flags = BindingFlags.Public | BindingFlags.NonPublic |
                        BindingFlags.Instance;
            foreach (var prop in GetPropertyValues(row, flags))
            {
                Console.Out.WriteLine(prop);
                Console.Out.Flush();
            }

            Console.Out.Flush();
            var prediction = predictionEngine.Predict(row);
            var debuggy = prediction;
            var debug = row;
            // Do something (print out Size property) with current Housing Data object being evaluated
            Console.WriteLine(row);
            Console.WriteLine(row);
        }
        /*
        // Get DataViewSchema of IDataView
        var columns = data.Schema;

        // Create DataViewCursor
        using (var cursor = data.GetRowCursor(columns))
        {
            // Define variables where extracted values will be stored to
            float size = default;
            VBuffer<float> historicalPrices = default;
            float currentPrice = default;

            // Define delegates for extracting values from columns
            var sizeDelegate = cursor.GetGetter<float>(columns[0]);
            var historicalPriceDelegate =
                cursor.GetGetter<VBuffer<float>>(columns[1]);
            var currentPriceDelegate = cursor.GetGetter<float>(columns[2]);

            // Iterate over each row
            while (cursor.MoveNext())
            {
                //Get values from respective columns
                sizeDelegate.Invoke(ref size);
                historicalPriceDelegate.Invoke(ref historicalPrices);
                currentPriceDelegate.Invoke(ref currentPrice);
            }
        }
        */
        //var dataExcerpt = dataView.GetDataExcerpt(labelColumnName);

        sb.AppendLine(
            "import pandas as pd\n" +
            "import matplotlib.pyplot as plt\n" +
            "from sklearn.inspection import plot_partial_dependence\n" +
            "\n" +
            "# Load the data\n" +
            $"data = pd.DataFrame({data})\n" +
            "\n" +
            "# Plot Ceteris Paribus for feature at index " + featureIndex +
            "\n" +
            "fig, ax = plt.subplots(figsize=(10, 6))\n" +
            $"plot_partial_dependence(model, data, features=[{featureIndex}], ax=ax)\n" +
            "plt.title('Ceteris Paribus Plot')\n" +
            "plt.show()");
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