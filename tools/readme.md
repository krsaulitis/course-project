Here are some of the tools required and the installation instructions for them.

### VISQOL
Clone - git@github.com:google/visqol.git
Follow the instruction manual in the repository to install the tool or...
- Install bazelisk using brew - `brew install bazelisk` (https://github.com/bazelbuild/bazelisk)
- Make sure that the Xcode command line tools are installed - `xcode-select --install` and license is accepted - `sudo xcodebuild -license`
- (optional) Provide xcode with correct directory - `sudo xcode-select -s /Applications/Xcode.app/Contents/Developer`
- Check available xcode sdks - `xcodebuild -showsdks`
- Run `bazel build :visqol -c opt --macos_sdk_version=14.0` in the visqol (cloned) directory
- ./tools/visqol/bazel-bin/visqol --reference_file ./datasets/ljspeech/wavs/LJ038-0050.wav --degraded_file ./inference/vits/samples/sample_1.wav --use_speech_mode --verbose
- ./bazel-bin/visqol --reference_file ../../datasets/ljspeech/wavs/LJ038-0050.wav --degraded_file ../../inference/vits/samples/sample_1.wav --use_speech_mode --verbose

### FFMPEG
- `ffmpeg -i ./datasets/ljspeech/wavs/LJ038-0050.wav -ar 16000 ./inference/output.wav`