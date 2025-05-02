namespace Italbytz.ML;

/// <summary>
///     Represents a mapping of category labels to their corresponding values.
/// </summary>
/// <remarks>
///     This class is used to define a mapping between category labels and their
///     corresponding values.
///     It is typically used in machine learning tasks where categorical data needs
///     to be represented as numerical values.
/// </remarks>
public class CategoryLookupMap
{
    /// <summary>
    ///     Gets or sets the value associated with the category.
    /// </summary>
    /// <value>A float value representing the category's value.</value>
    /// <remarks>
    ///     This property is typically used to represent the numerical value of the
    ///     category in the mapping.
    /// </remarks>
    public float Value { get; set; }

    /// <summary>
    ///     Gets or sets the category label.
    /// </summary>
    /// <value>A string representing the category label.</value>
    /// <remarks>
    ///     This property is typically used to represent the label of the category in
    ///     the mapping.
    /// </remarks>
    public required string Category { get; set; }
}