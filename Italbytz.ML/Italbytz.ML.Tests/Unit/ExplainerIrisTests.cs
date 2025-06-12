using Italbytz.ML.Data;
using Italbytz.ML.ModelBuilder.Configuration;
using JetBrains.Annotations;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace Italbytz.ML.Tests;

[TestClass]
[TestSubject(typeof(Explainer))]
public class ExplainerIrisTests
{
    private readonly Explainer _explainer;
    private readonly ITransformer? _irisModel;
    private readonly IDataView? _scoredIrisData;

    public ExplainerIrisTests()
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        // Load the model
        var modelPath =
            Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Models",
                "iris.mlnet");
        _irisModel = mlContext.Model.Load(modelPath, out _);
        // Load the raw data
        var dataPath = Path.Combine(AppDomain.CurrentDomain.BaseDirectory,
            "Data", "Iris.csv");
        var dataView = mlContext.Data.LoadFromTextFile<IrisModelInput>(
            dataPath, ',', true);
        // Transform the data
        _scoredIrisData = _irisModel.Transform(dataView);
        // Create the explainer
        _explainer = new Explainer(_irisModel, dataView,
            ScenarioType.Classification,
            "class");
    }

    [TestMethod]
    public void TestFeatureContribution()
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        var predictionTransformer = _irisModel.ExtractIPredictionTransformer();

        FeatureContributionCalculatingTransformer?
            featureContributionCalculator =
                null;
        if (predictionTransformer is
            ISingleFeaturePredictionTransformer<ICalculateFeatureContribution>
            compatiblePredictionTransformer)
            featureContributionCalculator = mlContext.Transforms
                .CalculateFeatureContribution(compatiblePredictionTransformer,
                    normalize: false).Fit(_scoredIrisData);
        Assert.IsNull(featureContributionCalculator);
    }

    [TestMethod]
    public void TestPfi()
    {
        var pfi = _explainer.GetPermutationFeatureImportanceTable(
            Metric.MacroAccuracy);
        Assert.IsNotNull(pfi);
        Assert.IsTrue(pfi.Contains("Feature, Importance"));
        Assert.IsTrue(pfi.Contains("petal length"));
        Assert.IsTrue(pfi.Contains("petal width"));
    }

    [TestMethod]
    public void TestCeterisParibusScript()
    {
        var script = _explainer
            .GetCeterisParibusScript<IrisModelInput, IrisModelOutput>();
        Assert.IsNotNull(script);
        Assert.IsTrue(script.Length > 0);
        // Additional checks can be added based on expected content of the script
    }

    private class IrisModelInput
    {
        [LoadColumn(0)]
        [ColumnName(@"sepal length")]
        public float Sepal_length { get; set; }

        [LoadColumn(1)]
        [ColumnName(@"sepal width")]
        public float Sepal_width { get; set; }

        [LoadColumn(2)]
        [ColumnName(@"petal length")]
        public float Petal_length { get; set; }

        [LoadColumn(3)]
        [ColumnName(@"petal width")]
        public float Petal_width { get; set; }

        [LoadColumn(4)] [ColumnName(@"class")] public string Class { get; set; }
    }

    private class IrisModelOutput
    {
        [ColumnName(@"sepal length")] public float Sepal_length { get; set; }

        [ColumnName(@"sepal width")] public float Sepal_width { get; set; }

        [ColumnName(@"petal length")] public float Petal_length { get; set; }

        [ColumnName(@"petal width")] public float Petal_width { get; set; }

        [ColumnName(@"class")] public uint Class { get; set; }

        [ColumnName(@"Features")] public float[] Features { get; set; }

        [ColumnName(@"PredictedLabel")]
        public string PredictedLabel { get; set; }

        [ColumnName(@"Score")] public float[] Score { get; set; }
    }
}