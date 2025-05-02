using Italbytz.Ports.Algorithms.AI.Learning.ML;

namespace Italbytz.ML;

/// <inheritdoc />
public class MulticlassClassificationInputSchema : ICustomMappingInputSchema
{
    /// <inheritdoc />
    public float[] Features { get; set; }
}