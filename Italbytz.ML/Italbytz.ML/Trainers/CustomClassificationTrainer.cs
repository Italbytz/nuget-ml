namespace Italbytz.ML.Trainers;

/// <inheritdoc />
public abstract class CustomClassificationTrainer<TOutput> : CustomTrainer<
    ClassificationInput,
    TOutput> where TOutput : class, new()
{
}