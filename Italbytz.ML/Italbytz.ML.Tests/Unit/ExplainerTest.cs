using Italbytz.ML.Data;
using Italbytz.ML.ModelBuilder.Configuration;
using Italbytz.ML.UCIMLR;
using JetBrains.Annotations;
using Microsoft.ML;

namespace Italbytz.ML.Tests;

[TestClass]
[TestSubject(typeof(Explainer))]
public class ExplainerTest
{
    [TestMethod]
    public void TestIris()
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        // Load the model
        var modelPath =
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models",
                "iris.mlnet");
        var mlModel = mlContext.Model.Load(modelPath, out _);
        // Load the raw data
        var dataPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory,
            "Data", "Iris.csv");
        var dataView = mlContext.Data.LoadFromTextFile<IrisModelInput>(
            dataPath, ',', true);
        // Transform the data
        dataView = mlModel.Transform(dataView);
        // Create the explainer
        var explainer = new Explainer(mlModel, dataView,
            ScenarioType.Classification,
            "class");
        var pfi = explainer.GetPermutationFeatureImportanceTable(
            Metric.MacroAccuracy);
        Console.WriteLine(pfi);
    }
}