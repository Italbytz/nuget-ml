namespace Italbytz.ML;

/// <summary>
///     Represents an excerpt of data used for machine learning operations.
/// </summary>
public interface IDataExcerpt
{
    /// <summary>
    ///     Gets the names of the features used in the dataset.
    /// </summary>
    /// <remarks>
    ///     Each string in the array corresponds to a feature and matches the positions
    ///     in the feature vectors stored in the Features collection.
    /// </remarks>
    public string[] FeatureNames { get; }

    /// <summary>
    ///     Gets the collection of labels, where each label corresponds to a data
    ///     point.
    /// </summary>
    /// <remarks>
    ///     Each label in the list represents the class or category of the
    ///     corresponding
    ///     feature vector at the same index in the Features collection.
    /// </remarks>
    public List<uint> Labels { get; }

    /// <summary>
    ///     Gets the collection of feature vectors, where each vector represents a data
    ///     point.
    /// </summary>
    /// <remarks>
    ///     Each float array in the list represents the feature values for a single
    ///     instance.
    /// </remarks>
    public List<float[]> Features { get; }

    /// <summary>
    ///     Gets the unique label values present in the dataset.
    /// </summary>
    public uint[] UniqueLabelValues { get; }

    /// <summary>
    ///     Retrieves the values of a specific feature as a column.
    /// </summary>
    /// <param name="featureName">The name of the feature to retrieve.</param>
    /// <returns>An array of floating-point values representing the feature column.</returns>
    public float[] GetFeatureColumn(string featureName);

    /// <summary>
    ///     Retrieves the unique values present in a specific feature.
    /// </summary>
    /// <param name="featureName">The name of the feature to analyze.</param>
    /// <returns>
    ///     An array of unique floating-point values present in the specified
    ///     feature.
    /// </returns>
    public float[] GetUniqueFeatureValues(string featureName);
}