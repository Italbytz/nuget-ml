namespace Italbytz.ML;

/// <summary>
/// Represents a set of metrics for evaluating machine learning models.
/// </summary>
public class Metrics
{
    /// <summary>
    /// Indicates whether the model is a binary classification model.
    /// </summary>
    public bool IsBinaryClassification { get; set; } = false;

    /// <summary>
    /// Indicates whether the model is a multiclass classification model.
    /// </summary>
    public bool IsMulticlassClassification { get; set; } = false;

    /// <summary>
    /// Gets or sets the macro-averaged accuracy of the model.
    /// </summary>
    public double MacroAccuracy { get; set; }

    /// <summary>
    /// Gets or sets the overall accuracy of the model.
    /// </summary>
    public double Accuracy { get; set; }

    /// <summary>
    /// Gets or sets the area under the ROC curve (AUC-ROC) for the model.
    /// </summary>
    public double AreaUnderRocCurve { get; set; }

    /// <summary>
    /// Gets or sets the F1 score of the model.
    /// </summary>
    public double F1Score { get; set; }

    /// <summary>
    /// Gets or sets the area under the precision-recall curve (AUC-PR) for the model.
    /// </summary>
    public double AreaUnderPrecisionRecallCurve { get; set; }
}
