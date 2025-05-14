using Microsoft.ML.Data;

namespace Italbytz.ML;

/// <summary>
///     Interface that defines the schema for multiclass classification output in
///     machine learning models.
/// </summary>
/// <remarks>
///     This interface specifies properties that should be returned in
///     the output of multiclass classification machine learning operations.
/// </remarks>
public interface ICustomMappingMulticlassClassificationOutputSchema
{
    /// <summary>
    ///     Gets or sets the predicted class label.
    /// </summary>
    /// <value>A non-negative integer representing the predicted class.</value>
    /// <remarks>
    ///     This property is typically used to represent the predicted class label for
    ///     multiclass classification tasks.
    ///     The value should correspond to one of the classes defined in the model's
    ///     training data.
    /// </remarks>
    public uint PredictedLabel { get; set; }

    /// <summary>
    ///     Gets or sets the raw scores produced by the model for each class.
    /// </summary>
    /// <value>A vector of float values representing the scores for each class.</value>
    /// <remarks>
    ///     This property is typically used to represent the raw scores for each class
    ///     in multiclass classification tasks.
    ///     The vector should have a length equal to the number of classes defined in
    ///     the model's training data.
    ///     The scores can be used for ranking or threshold comparisons.
    /// </remarks>
    public VBuffer<float> Score { get; set; }

    /// <summary>
    ///     Gets or sets the probabilities of the instance belonging to each class.
    /// </summary>
    /// <value>A vector of float values representing the probabilities for each class.</value>
    /// <remarks>
    ///     This property is typically used to represent the probabilities for each
    ///     class in multiclass classification tasks.
    ///     The vector should have a length equal to the number of classes defined in
    ///     the model's training data.
    ///     The probabilities should sum to 1 across all classes.
    /// </remarks>
    public VBuffer<float> Probability { get; set; }
}