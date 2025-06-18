using Microsoft.ML.Data;

namespace Italbytz.ML.Data;

/// <summary>
///     Extension methods for the <see cref="MulticlassClassificationMetrics" />
///     class to calculate
///     additional metrics.
/// </summary>
public static class MulticlassClassificationMetricsExtensions
{
    /// <summary>
    ///     Calculates the F1 score for binary classification.
    /// </summary>
    /// <param name="positiveClassIndex">
    ///     The index of the positive class in the
    ///     confusion matrix (default is 1).
    /// </param>
    /// <returns>The F1 score as a value between 0 and 1.</returns>
    public static double F1ScoreBinary(
        this MulticlassClassificationMetrics metrics,
        int positiveClassIndex = 1)
    {
        return metrics.ConfusionMatrix.F1ScoreBinary(positiveClassIndex);
    }

    /// <summary>
    ///     Calculates the macro-averaged F1 score across all classes in the given
    ///     confusion matrix.
    ///     This treats each class equally by averaging the F1 scores of all classes,
    ///     regardless of their support.
    /// </summary>
    public static double F1Macro(
        this MulticlassClassificationMetrics metrics)
    {
        return metrics.ConfusionMatrix.F1Macro();
    }
}