namespace Italbytz.ML.ModelBuilder.Configuration;

public class FileValidationOptionV0 : MBConfig, IValidationOption
{
    public override int Version => 0;
    public override string Type => "FileValidationOption";
    public string? FilePath { get; set; }
}