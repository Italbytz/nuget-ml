using Microsoft.ML;
using Microsoft.ML.Transforms;

namespace Italbytz.ML.Trainers;

/// <summary>
///     Base class for custom trainers that use a custom mapping estimator.
/// </summary>
public abstract class
    CustomTrainer<TInput, TOutput> : IEstimator<ITransformer>
    where TOutput : class, new()
    where TInput : class, new()
{
    /// <inheritdoc />
    public SchemaShape GetOutputSchema(SchemaShape inputSchema)
    {
        return GetCustomMappingEstimator().GetOutputSchema(inputSchema);
    }

    /// <inheritdoc />
    public ITransformer Fit(IDataView input)
    {
        PrepareForFit(input);
        return GetCustomMappingEstimator().Fit(input);
    }

    /// <summary>
    ///     Prepares the input data for the training process. This method is called
    ///     before fitting the custom mapping estimator and should be implemented
    ///     by derived classes to perform any necessary preprocessing or setup.
    /// </summary>
    /// <param name="input">The input data view to be prepared.</param>
    protected void PrepareForFit(IDataView input)
    {
    }

    /// <summary>
    ///     Gets the custom mapping estimator used for transforming the input data
    ///     into the desired output format. This method should be implemented by
    ///     derived classes to provide the specific mapping logic.
    /// </summary>
    /// <returns>
    ///     A <see cref="CustomMappingEstimator{TInput, TOutput}" /> instance
    ///     that defines the mapping logic.
    /// </returns>
    private CustomMappingEstimator<TInput, TOutput>
        GetCustomMappingEstimator()
    {
        var mlContext = ThreadSafeMLContext.LocalMLContext;
        return mlContext.Transforms
            .CustomMapping(
                GetMapping(), null);
    }

    private Action<TInput, TOutput> GetMapping()
    {
        return Map;
    }

    /// <summary>
    ///     Mapping method that defines how to transform the input data into the
    ///     output data.
    /// </summary>
    /// <param name="input">Input data</param>
    /// <param name="output">Output data</param>
    protected abstract void Map(TInput input, TOutput output);
}