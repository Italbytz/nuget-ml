namespace Italbytz.ML;

/// <summary>
///     Interface that defines the schema for custom mapping input in machine
///     learning models.
/// </summary>
public interface ICustomMappingInputSchema
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