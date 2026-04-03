# Ice Queen 

Ice Queen is a tool that improves the quality of old, corrupted or low-quality audio files. The only currently supported format is .wav. 

# Usage guide 
I strongly recommend you use <a href="https://github.com/astral-sh/uv"> uv </a>. Once installed, run 
```console
  uv sync
```
to install dependencies. 

Otherwise, you need to install all dependencies, listed in pyproject.toml, using your package manager of choice. 

Once the environment is set up, you can repair a corrupted file like this:   
```console
  uv run app.py /path_to_source_file/sounds.wav /path_to_repaired_file/sounds_fixed.py
```

# Samples
For a few samples of what Ice-Queen can do, see /assets. 
(NOTE: the "something-denoised-2" files are recovered using a suboptimal method, and as such should be ignored)
