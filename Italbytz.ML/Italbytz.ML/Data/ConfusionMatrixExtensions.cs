using Microsoft.ML.Data;

namespace Italbytz.ML.Data;

/// <summary>
///     Extension methods for the <see cref="ConfusionMatrix" /> class to calculate
///     additional metrics.
/// </summary>
public static class ConfusionMatrixExtensions
{
    /// <summary>
    ///     Calculates the F1 score for binary classification.
    /// </summary>
    /// <param name="positiveClassIndex">
    ///     The index of the positive class in the
    ///     confusion matrix (default is 1).
    /// </param>
    /// <returns>The F1 score as a value between 0 and 1.</returns>
    public static double F1ScoreBinary(this ConfusionMatrix matrix,
        int positiveClassIndex = 1)
    {
        var precision = matrix.PerClassPrecision[positiveClassIndex];
        var recall = matrix.PerClassRecall[positiveClassIndex];

        if (precision + recall == 0) return 0;

        return 2 * (precision * recall) / (precision + recall);
    }
}