using System.Text.Json.Serialization;

namespace Italbytz.ML.ModelBuilder.Configuration;

[JsonDerivedType(typeof(RegressionTrainingOptionV2))]
[JsonDerivedType(typeof(ClassificationTrainingOptionV2))]
public interface ITrainingOption
{
    int TrainingTime { get; set; }

    int? Seed { get; set; }

    string? OutputFolder { get; set; }

    IValidationOption? ValidationOption { get; set; }
}