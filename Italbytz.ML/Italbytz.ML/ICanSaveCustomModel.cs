namespace Italbytz.ML;

/// <summary>
///     Provides the ability to persist a trained ML model to a filesystem path.
///     Implementations are responsible for serializing model artifacts and any
///     associated metadata required for later loading.
/// </summary>
public interface ICanSaveCustomModel
{
    /// <summary>
    ///     Saves the trained ML model and any required metadata to the provided
    ///     stream.
    ///     Implementations should serialize all artifacts necessary for later loading.
    /// </summary>
    /// <param name="stream">A writable stream where the model will be persisted.</param>
    void Save(Stream stream);
}