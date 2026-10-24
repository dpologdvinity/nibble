# Security policy

## Scope

nibble loads model files (GGUF) and prompts supplied by the user and runs them on the local machine. The main
attack surface is **parsing untrusted model files**: the GGUF reader, the tokenizer construction and the tensor
binding. A malformed file must produce a `nibble::Error`, never memory corruption, an infinite loop, or an
allocation far larger than the file. nibble has no network code and no scripting engine.

Examples of reportable problems:

* a crash, sanitizer report, hang or out-of-bounds read triggered by a crafted `.gguf` file or prompt;
* memory use that grows much faster than the size of the input file;
* text from a file reaching the terminal or an error message without escaping control characters.

Wrong or low-quality model output is not a security issue.

## Reporting

Please report vulnerabilities privately through GitHub's **Report a vulnerability** form (Security tab →
*Advisories*) on this repository rather than in a public issue. Include the command line, the nibble commit
(`nibble bench --version`), and a minimal file that reproduces the problem (or a hex dump of one). You will get an
acknowledgement and a fix or an explanation; please allow some time before public disclosure.

## Hardening already in place

* All counts and sizes in the file are validated against the file length with overflow-safe arithmetic before use.
* Debug builds run under AddressSanitizer and UndefinedBehaviorSanitizer (including `float-cast-overflow`) in CI, and a
  libFuzzer target exercises the parser and loader on every push.
* The decode loop allocates nothing; memory-amplification and out-of-memory behaviour are covered by tests.
* `tools/convert_hf.sh` verifies downloaded files against Hugging Face's published hashes and refuses repositories whose
  configuration could make the converter run repository code.

## Supported versions

Only the latest commit on `master` is supported.
