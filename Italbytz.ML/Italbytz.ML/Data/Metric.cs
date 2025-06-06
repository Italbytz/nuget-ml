namespace Italbytz.ML.Data;

/// <summary>
/// Specifies the different evaluation metrics that can be used to assess the performance of machine learning models.
/// </summary>
/// <remarks>
/// The <see cref="Metric"/> enumeration includes metrics for classification, regression, and ranking tasks.
/// </remarks>
public enum Metric
{
    Accuracy,
    AreaUnderRocCurve,
    AreaUnderPrecisionRecallCurve,
    F1Score,
    LogLoss,
    MeanAbsoluteError,
    MeanSquaredError,
    RootMeanSquaredError,
    R2Score,
    SpearmanCorrelationCoefficient,
    MacroAccuracy,
    MicroAccuracy
}