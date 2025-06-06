using System.Text;
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

    /// <summary>
    ///     Generates a Python script using scikit-learn and matplotlib to plot a
    ///     confusion matrix
    ///     based on the data contained in the given <see cref="ConfusionMatrix" />.
    ///     The script reconstructs the actual and predicted label arrays from the
    ///     matrix counts,
    ///     creates a confusion matrix plot, and saves it to the specified file.
    /// </summary>
    /// <param name="plotFileName">
    ///     The filename (including path) where the generated
    ///     plot image will be saved.
    /// </param>
    /// <returns>
    ///     A string containing the Python script that, when executed, will generate
    ///     and save the confusion matrix plot.
    /// </returns>
    public static string SklearnScript(this ConfusionMatrix matrix,
        string plotFileName)
    {
        var sb = new StringBuilder();
        sb.AppendLine("import matplotlib.pyplot as plt");
        sb.AppendLine("from sklearn import metrics");
        var actual = new List<int>();
        var predicted = new List<int>();
        var rowIndex = 0;
        foreach (var row in matrix.Counts)
        {
            var columnIndex = 0;
            foreach (var count in row)
            {
                for (var i = 0; i < count; i++)
                {
                    actual.Add(columnIndex);
                    predicted.Add(rowIndex);
                }

                columnIndex++;
            }

            rowIndex++;
        }

        var actualArray = string.Join(", ", actual);
        var predictedArray = string.Join(", ", predicted);
        sb.AppendLine($"actual = [{actualArray}]");
        sb.AppendLine($"predicted = [{predictedArray}]");
        sb.AppendLine(
            "confusion_matrix = metrics.confusion_matrix(actual, predicted)");
        sb.AppendLine(
            "cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix)");
        sb.AppendLine("cm_display.plot()");
        sb.AppendLine($"plt.savefig('{plotFileName}')");
        return sb.ToString();
    }
}