namespace Italbytz.ML;

/// <summary>
///     Provides the ability to persist a trained ML model to a filesystem path.
///     Implementations are responsible for serializing model artifacts and any
///     associated metadata required for later loading.
/// </summary>
public interface ICanSaveModelToPath
{
    /// <summary>
    ///     Saves the model to the specified file system path.
    ///     The implementation should create or overwrite the target file(s)
    ///     and ensure all necessary resources are written so the model can be
    ///     reloaded later.
    /// </summary>
    /// <param name="modelPath">
    ///     A file system path or directory where the model will be
    ///     saved.
    /// </param>
    void Save(string modelPath);
}