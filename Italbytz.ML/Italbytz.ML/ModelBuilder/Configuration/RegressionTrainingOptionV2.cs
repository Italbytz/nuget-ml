namespace Italbytz.ML.ModelBuilder.Configuration;

public class RegressionTrainingOptionV2 : MBConfig, ITrainingOption
{
    public override int Version => 2;
    public override string? Type => "RegressionTrainingOption";
    public int? MaxModelToExplore { get; set; }
    public int? MaximumMemoryToUseInMB { get; set; }
    public bool Subsampling { get; set; } = false;
    public string? LabelColumn { get; set; }
    public string[]? AvailableTrainers { get; set; }
    public string? Tuner { get; set; }
    public string? OptimizeMetric { get; set; }
    public int TrainingTime { get; set; }
    public int? Seed { get; set; }
    public string? OutputFolder { get; set; }
    public IValidationOption? ValidationOption { get; set; }
}