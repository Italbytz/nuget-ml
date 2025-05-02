using Italbytz.Ports.Algorithms.AI.Learning.ML;
using Microsoft.ML.Data;

namespace Italbytz.ML;

/// <inheritdoc />
public class
    TernaryClassificationClassificationOutputSchema :
    ICustomMappingMulticlassClassificationOutputSchema
{
    /// <inheritdoc />
    public uint PredictedLabel { get; set; }

    /// <inheritdoc />
    [VectorType(3)]
    public VBuffer<float> Score { get; set; }

    /// <inheritdoc />
    public VBuffer<float> Probability { get; set; }
}

/// <inheritdoc />
public class
    QuaternaryClassificationClassificationOutputSchema :
    ICustomMappingMulticlassClassificationOutputSchema
{
    /// <inheritdoc />
    public uint PredictedLabel { get; set; }

    /// <inheritdoc />
    [VectorType(4)]
    public VBuffer<float> Score { get; set; }

    /// <inheritdoc />
    public VBuffer<float> Probability { get; set; }
}

/// <inheritdoc />
public class
    QuinaryClassificationClassificationOutputSchema :
    ICustomMappingMulticlassClassificationOutputSchema
{
    /// <inheritdoc />
    public uint PredictedLabel { get; set; }

    /// <inheritdoc />
    [VectorType(5)]
    public VBuffer<float> Score { get; set; }

    /// <inheritdoc />
    public VBuffer<float> Probability { get; set; }
}

/// <inheritdoc />
public class
    SenaryClassificationClassificationOutputSchema :
    ICustomMappingMulticlassClassificationOutputSchema
{
    /// <inheritdoc />
    public uint PredictedLabel { get; set; }

    /// <inheritdoc />
    [VectorType(6)]
    public VBuffer<float> Score { get; set; }

    /// <inheritdoc />
    public VBuffer<float> Probability { get; set; }
}

/// <inheritdoc />
public class
    SeptenaryClassificationClassificationOutputSchema :
    ICustomMappingMulticlassClassificationOutputSchema
{
    /// <inheritdoc />
    public uint PredictedLabel { get; set; }

    /// <inheritdoc />
    [VectorType(7)]
    public VBuffer<float> Score { get; set; }

    /// <inheritdoc />
    public VBuffer<float> Probability { get; set; }
}

/// <inheritdoc />
public class
    OctonaryClassificationClassificationOutputSchema :
    ICustomMappingMulticlassClassificationOutputSchema
{
    /// <inheritdoc />
    public uint PredictedLabel { get; set; }

    /// <inheritdoc />
    [VectorType(8)]
    public VBuffer<float> Score { get; set; }

    /// <inheritdoc />
    public VBuffer<float> Probability { get; set; }
}

/// <inheritdoc />
public class
    NonaryClassificationClassificationOutputSchema :
    ICustomMappingMulticlassClassificationOutputSchema
{
    /// <inheritdoc />
    public uint PredictedLabel { get; set; }

    /// <inheritdoc />
    [VectorType(9)]
    public VBuffer<float> Score { get; set; }

    /// <inheritdoc />
    public VBuffer<float> Probability { get; set; }
}

/// <inheritdoc />
public class
    DenaryClassificationClassificationOutputSchema :
    ICustomMappingMulticlassClassificationOutputSchema
{
    /// <inheritdoc />
    public uint PredictedLabel { get; set; }

    /// <inheritdoc />
    [VectorType(10)]
    public VBuffer<float> Score { get; set; }

    /// <inheritdoc />
    public VBuffer<float> Probability { get; set; }
}

/// <inheritdoc />
public class
    UndenaryClassificationClassificationOutputSchema :
    ICustomMappingMulticlassClassificationOutputSchema
{
    /// <inheritdoc />
    public uint PredictedLabel { get; set; }

    /// <inheritdoc />
    [VectorType(11)]
    public VBuffer<float> Score { get; set; }

    /// <inheritdoc />
    public VBuffer<float> Probability { get; set; }
}

/// <inheritdoc />
public class
    DuodenaryClassificationClassificationOutputSchema :
    ICustomMappingMulticlassClassificationOutputSchema
{
    /// <inheritdoc />
    public uint PredictedLabel { get; set; }

    /// <inheritdoc />
    [VectorType(12)]
    public VBuffer<float> Score { get; set; }

    /// <inheritdoc />
    public VBuffer<float> Probability { get; set; }
}

/// <inheritdoc />
public class
    TridenaryClassificationClassificationOutputSchema :
    ICustomMappingMulticlassClassificationOutputSchema
{
    /// <inheritdoc />
    public uint PredictedLabel { get; set; }

    /// <inheritdoc />
    [VectorType(13)]
    public VBuffer<float> Score { get; set; }

    /// <inheritdoc />
    public VBuffer<float> Probability { get; set; }
}

/// <inheritdoc />
public class
    TetradenaryClassificationClassificationOutputSchema :
    ICustomMappingMulticlassClassificationOutputSchema
{
    /// <inheritdoc />
    public uint PredictedLabel { get; set; }

    /// <inheritdoc />
    [VectorType(14)]
    public VBuffer<float> Score { get; set; }

    /// <inheritdoc />
    public VBuffer<float> Probability { get; set; }
}

/// <inheritdoc />
public class
    PentadenaryClassificationClassificationOutputSchema :
    ICustomMappingMulticlassClassificationOutputSchema
{
    /// <inheritdoc />
    public uint PredictedLabel { get; set; }

    /// <inheritdoc />
    [VectorType(15)]
    public VBuffer<float> Score { get; set; }

    /// <inheritdoc />
    public VBuffer<float> Probability { get; set; }
}