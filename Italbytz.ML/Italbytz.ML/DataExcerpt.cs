using System.Collections.Immutable;

namespace Italbytz.ML;

/// <inheritdoc />
public class DataExcerpt : IDataExcerpt
{
    private readonly Dictionary<string, float[]> _uniqueFeatureValues = new();
    private readonly Dictionary<string, float[]> _featureColumns = new();
    private uint[]? _uniqueLabelValues;

    /// <summary>
    ///     Initializes a new instance of the <see cref="DataExcerpt" /> class.
    /// </summary>
    /// <param name="features">
    ///     List of feature arrays where each array represents a
    ///     data point.
    /// </param>
    /// <param name="featureNames">
    ///     Array of feature names corresponding to the
    ///     features.
    /// </param>
    /// <param name="labels">List of labels associated with each data point.</param>
    public DataExcerpt(List<float[]> features,
        ImmutableArray<ReadOnlyMemory<char>> featureNames, List<uint> labels)
    {
        Features = features;
        FeatureNames = featureNames.Select(f => f.ToString()).ToArray();
        Labels = labels;
    }

    /// <inheritdoc />
    public string[] FeatureNames { get; }

    /// <inheritdoc />
    public List<uint> Labels { get; }

    /// <inheritdoc />
    public List<float[]> Features { get; }

    /// <inheritdoc />
    public uint[] UniqueLabelValues
    {
        get
        {
            if (_uniqueLabelValues != null) return _uniqueLabelValues;
            var unique = new HashSet<uint>(Labels);
            _uniqueLabelValues = unique.ToArray();

            return _uniqueLabelValues;
        }
    }

    /// <inheritdoc />
    public float[] GetFeatureColumn(string featureName)
    {
        if (_featureColumns.TryGetValue(featureName, out var column))
            return column;
        var index = Array.IndexOf(FeatureNames, featureName);
        if (index < 0)
            throw new ArgumentException($"Feature '{featureName}' not found.");
        column = Features.Select(f => f[index]).ToArray();
        _featureColumns[featureName] = column;
        return column;
    }

    /// <inheritdoc />
    public float[] GetUniqueFeatureValues(string featureName)
    {
        var column = GetFeatureColumn(featureName);
        if (_uniqueFeatureValues.TryGetValue(featureName, out var uniqueValues))
            return uniqueValues;
        var unique = new HashSet<float>(column);
        uniqueValues = unique.ToArray();
        _uniqueFeatureValues[featureName] = uniqueValues;
        return uniqueValues;
    }
}