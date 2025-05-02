using Italbytz.Ports.Algorithms.AI.Learning.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML;

/// <inheritdoc cref="ICustomMappingBinaryClassificationOutputSchema" />
public class
    BinaryClassificationOutputSchema :
    ICustomMappingBinaryClassificationOutputSchema
{
    /// <inheritdoc />
    [KeyType(2)]
    public uint PredictedLabel { get; set; }

    /// <inheritdoc />
    public float Score { get; set; }

    /// <inheritdoc />
    public float Probability { get; set; }
}