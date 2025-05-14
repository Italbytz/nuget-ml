namespace Italbytz.ML;

/// <summary>
///     Interface that defines the schema for binary classification output in
///     machine learning models.
/// </summary>
/// <remarks>
///     This interface specifies properties that should be returned in
///     the output of binary classification machine learning operations.
/// </remarks>
public interface ICustomMappingBinaryClassificationOutputSchema
{
    /// <summary>
    ///     Gets or sets the predicted class label.
    /// </summary>
    /// <value>
    ///     A non-negative integer representing the predicted class (typically 0 or
    ///     1 or 1 or 2 for binary classification).
    /// </value>
    public uint PredictedLabel { get; set; }

    /// <summary>
    ///     Gets or sets the raw score produced by the model for the positive class.
    /// </summary>
    /// <value>The score value, typically used for ranking or threshold comparisons.</value>
    public float Score { get; set; }

    /// <summary>
    ///     Gets or sets the probability of the instance belonging to the positive
    ///     class.
    /// </summary>
    /// <value>A value between 0 and 1 representing the probability.</value>
    public float Probability { get; set; }
}