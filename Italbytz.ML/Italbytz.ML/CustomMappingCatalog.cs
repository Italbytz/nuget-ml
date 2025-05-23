using Microsoft.ML.Data;

namespace Italbytz.ML;

#region Columns

/// <summary>
///     Interface that defines a score as output for custom mapping in
///     machine learning models.
/// </summary>
public interface IScoreScalar
{
    /// <summary>
    ///     Gets or sets the raw score produced by the model for the positive class.
    /// </summary>
    /// <value>The score value, typically used for ranking or threshold comparisons.</value>
    public float Score { get; set; }
}

/// <summary>
///     Interface that defines scores as output for custom mapping in
///     machine learning models.
/// </summary>
public interface IScoreVector
{
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
}

/// <summary>
///     Interface that defines a probability as output for custom mapping in
///     machine learning models.
/// </summary>
public interface IProbabilityScalar
{
    /// <summary>
    ///     Gets or sets the probability of the instance belonging to the positive
    ///     class.
    /// </summary>
    /// <value>A value between 0 and 1 representing the probability.</value>
    public float Probability { get; set; }
}

/// <summary>
///     Interface that defines probabilities as output for custom mapping in
///     machine learning models.
/// </summary>
public interface IProbabilityVector
{
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

/// <summary>
///     Interface that defines features as input for custom mapping in
///     machine learning models.
/// </summary>
public interface IFeatures
{
    /// <summary>
    ///     Gets or sets the features of the input data.
    /// </summary>
    /// <value>An array of float values representing the features.</value>
    /// <remarks>
    ///     This property is typically used to represent the input features for machine
    ///     learning models.
    ///     The array should be of a fixed size that matches the model's expected input
    ///     size.
    /// </remarks>
    public float[] Features { get; set; }
}

/// <summary>
///     Interface that defines PredictedLabel as output for custom mapping in
///     machine learning models.
/// </summary>
public interface IPredictedLabel
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
}

#endregion

#region Input

/// <summary>
///     Interface that defines the schema for custom mapping input in machine
///     learning models.
/// </summary>
public interface ICustomMappingInput : IFeatures
{
}

/// <inheritdoc />
public class CustomMappingInput : ICustomMappingInput
{
    /// <inheritdoc />
    public float[] Features { get; set; }
}

/// <inheritdoc />
public class ClassificationInput : CustomMappingInput
{
}

/// <inheritdoc />
public class RegressionInput : CustomMappingInput
{
}

#endregion

#region Output

/// <summary>
///     Interface that defines the output for custom mapping input in machine
///     learning models.
/// </summary>
public interface ICustomMappingOutput
{
}

/// <summary>
///     Interface that defines the schema for custom mapping output in machine
///     learning models.
/// </summary>
public interface IRegressionOutput : ICustomMappingOutput,
    IScoreScalar
{
}

/// <summary>
///     Interface that defines the schema for custom mapping output in machine
///     learning models.
/// </summary>
public interface IClassificationOutput : ICustomMappingOutput,
    IPredictedLabel
{
}

/// <summary>
///     Interface that defines the schema for binary classification output in
///     machine learning models.
/// </summary>
/// <remarks>
///     This interface specifies properties that should be returned in
///     the output of binary classification machine learning operations.
/// </remarks>
public interface IBinaryClassificationOutput : IClassificationOutput,
    IScoreScalar, IProbabilityScalar
{
}

/// <summary>
///     Interface that defines the schema for multiclass classification output in
///     machine learning models.
/// </summary>
/// <remarks>
///     This interface specifies properties that should be returned in
///     the output of multiclass classification machine learning operations.
/// </remarks>
public interface
    IMulticlassClassificationOutput : IClassificationOutput, IScoreVector,
    IProbabilityVector
{
}

/// <inheritdoc />
public class RegressionOutput : IRegressionOutput
{
    /// <inheritdoc />
    public float Score { get; set; }
}

/// <inheritdoc cref="IBinaryClassificationOutput" />
public class
    BinaryClassificationOutput :
    IBinaryClassificationOutput
{
    /// <inheritdoc />
    [KeyType(2)]
    public uint PredictedLabel { get; set; }

    /// <inheritdoc />
    public float Score { get; set; }

    /// <inheritdoc />
    public float Probability { get; set; }
}

public class MulticlassClassificationOutput :
    IMulticlassClassificationOutput
{
    /// <inheritdoc />
    public uint PredictedLabel { get; set; }

    /// <inheritdoc />
    [VectorType(2)]
    public virtual VBuffer<float> Score { get; set; }

    /// <inheritdoc />
    public VBuffer<float> Probability { get; set; }
}

/// <inheritdoc />
public class
    TernaryClassificationOutput :
    MulticlassClassificationOutput
{
    [VectorType(3)] public override VBuffer<float> Score { get; set; }
}

/// <inheritdoc />
public class
    QuaternaryClassificationOutput :
    MulticlassClassificationOutput
{
    [VectorType(4)] public override VBuffer<float> Score { get; set; }
}

/// <inheritdoc />
public class
    QuinaryClassificationOutput :
    MulticlassClassificationOutput
{
    [VectorType(5)] public override VBuffer<float> Score { get; set; }
}

/// <inheritdoc />
public class
    SenaryClassificationOutput :
    MulticlassClassificationOutput
{
    [VectorType(6)] public override VBuffer<float> Score { get; set; }
}

/// <inheritdoc />
public class
    SeptenaryClassificationOutput :
    MulticlassClassificationOutput
{
    [VectorType(7)] public override VBuffer<float> Score { get; set; }
}

/// <inheritdoc />
public class
    OctonaryClassificationOutput :
    MulticlassClassificationOutput
{
    [VectorType(8)] public override VBuffer<float> Score { get; set; }
}

/// <inheritdoc />
public class
    NonaryClassificationOutput :
    MulticlassClassificationOutput
{
    [VectorType(9)] public override VBuffer<float> Score { get; set; }
}

/// <inheritdoc />
public class
    DenaryClassificationOutput :
    MulticlassClassificationOutput
{
    [VectorType(10)] public override VBuffer<float> Score { get; set; }
}

/// <inheritdoc />
public class
    UndenaryClassificationOutput :
    MulticlassClassificationOutput
{
    [VectorType(11)] public override VBuffer<float> Score { get; set; }
}

/// <inheritdoc />
public class
    DuodenaryClassificationOutput :
    MulticlassClassificationOutput
{
    [VectorType(12)] public override VBuffer<float> Score { get; set; }
}

/// <inheritdoc />
public class
    TridenaryClassificationOutput :
    MulticlassClassificationOutput
{
    [VectorType(13)] public override VBuffer<float> Score { get; set; }
}

/// <inheritdoc />
public class
    TetradenaryClassificationOutput :
    MulticlassClassificationOutput
{
    [VectorType(14)] public override VBuffer<float> Score { get; set; }
}

/// <inheritdoc />
public class
    PentadenaryClassificationOutput :
    MulticlassClassificationOutput
{
    [VectorType(15)] public override VBuffer<float> Score { get; set; }
}

#endregion