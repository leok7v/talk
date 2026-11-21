# talk

A voice chat on one machine: press the microphone, speak, hear the
answer. Three small models run in one process, in plain C. The build
has no dependency but the C library, with Accelerate on a Mac and
OpenBLAS on Linux.

| part | model | file | size |
|---|---|---|---|
| speech to text | Cactus Whistle, 7 languages | `whistle.c` | 17 MB |
| chat | Falcon-H1 Tiny 90M | `falcon.c` | 116 MB |
| text to speech | Supertonic 3 | `supertonic.c` | 112 MB |
| speech to text, for a compare | Moonshine tiny, English | `moonshine.c` | 108 MB |
| chat, an experiment | Qwen3.5 0.8B | `qwen.c` | 812 MB |

## Models

    ./download.sh

The script asks where the model files go, then asks one time and
downloads the 4 models of the voice chat, about 355 MB. One more
question is for the Qwen chat model, 812 MB.

After a yes to the first question the files go under `huggingface.co`
in the home directory. The path is the user and the repo of the
model's URL:

    ~/huggingface.co/leok7v/whistle/whistle.cact
    ~/huggingface.co/leok7v/Falcon-H1-Tiny-Tool-Calling-90M-GGUF/Falcon-H1-Tiny-90M-Tool-Calling-Q8_0.gguf
    ~/huggingface.co/leok7v/supertonic/supertonic-q8.safetensors
    ~/huggingface.co/leok7v/moonshine-tiny/model.safetensors
    ~/huggingface.co/leok7v/Qwen3.5-0.8B-Astra-GGUF/Qwen3.5-0.8B-Astra-Q8_0.gguf

After a no the files go under `models` in the current directory, in
the same form, and nothing outside the directory changes:

    models/leok7v/whistle/whistle.cact

The programs look in `models` first, then in the home directory. A
download with no question:

    ./download.sh models -y

The word `models` is the directory, `-y` is a yes to the 4 models,
and `-a` takes the Qwen model too.

The model files are in these repos:

- https://huggingface.co/leok7v/whistle
- https://huggingface.co/leok7v/Falcon-H1-Tiny-Tool-Calling-90M-GGUF
- https://huggingface.co/leok7v/supertonic
- https://huggingface.co/leok7v/moonshine-tiny
- https://huggingface.co/leok7v/Qwen3.5-0.8B-Astra-GGUF

Run the server with `llm=qwen` for the Qwen model, or pick the engine
on the page.

## Build and run

    make
    ./talk

The build needs clang. On a Mac the build needs nothing more. On
Linux the build needs OpenBLAS with its headers. With ROCm at
the path `ROCM` of `Makefile`, the build adds the backend `hip` for
an AMD GPU.

Open http://127.0.0.1:8081/ in Safari, not in Chrome:

    open -a Safari http://127.0.0.1:8081/

Safari asks for the microphone at the first press of the button.
Allow it. The text box works with no microphone. The page picks the
language and the speech to text engine, or a compare of all engines.
Two switches set the noise suppression and auto gain of the browser.

Speak close to the microphone. Both speech to text models read a
close voice right and fail on a voice that is far or quiet.

`make` also builds one program per engine, for a test of the engine
alone:

    ./whistle wav=clip.wav
    ./moonshine wav=clip.wav
    ./falcon text="What is the capital of France?"
    ./qwen text="Why is the sky blue?" think=2 script='S{1,2}'
    ./supertonic text="Hello world." voice=F1 out=hello.wav

Options are `key=value` words. An unknown option stops the program
and prints the known options.

| option | default | meaning |
|---|---|---|
| `weights=` | per engine | `float`, or `packed`: the weights stay at 2, 4 and 8 bits |
| `gemm=` | `blas` | the backend: `blas`, `cpu`, or `metal` or `hip` for a GPU |
| `asr=` `llm=` `tts=` | the first installed | the engine of a role |
| `context=` | `16384` | the context of the chat model, in tokens |
| `state=` | none | a file that keeps a Falcon chat between runs |
| `temperature=` | `0.6` | the sampler; 0 is the greedy decode |
| `top_k=` `top_p=` | `20` `0.95` | the sampler |
| `presence=` `dry=` | `0` `0.8` | the penalties for a repeat |
| `seed=` | `0` | the same seed gives the same text; 0 is the time |
| `script=` | `*` | the form of a reply, see Scripts |
| `think=` | `0` | the sentences of thought of Qwen; 0 is no thought |
| `shift=` | `none` | a full context: `none`, `soft` or `exact` |
| `tokens=` | `256` | the most tokens of a reply |
| `beam=` | `5` | the beams of the speech to text decoder, 1 to 8 |
| `depth=` | `8` | the layers of the Whistle decoder, 2 to 8; fewer is faster |
| `save=` | none | a file that keeps each upload of the page |
| `page=` | `index.html` | the page file the server sends |
| `model=` | the engine's own | the model file, for an engine program |

The speech engine takes the model file with `model=`: the fp32, q8 or
q4 file of leok7v/supertonic. The q8 file is the default.

Falcon uses packed weights with no option, and the other engines use
float weights. Packed weights of Falcon feed and reply as fast as
float weights. On an Apple M3 the server holds 249 MB after one chat.
With `weights=float` it holds 593 MB, and with `weights=packed` 156
MB. Whistle is 3 times slower with packed weights on an Intel
i7-3667U.

## Scripts

A script gives a chat reply its form. The sampler checks each token
against the script, so the model cannot leave the form.

    S{1,3}                        1 to 3 plain sentences
    "The answer is " S            fixed words, then one sentence
    ("Yes"|"No") "."              a choice of fixed texts
    "{\"city\": \"" T "\"}"       a text with no quote, up to the fixed text
    "{\"seconds\": " N "}"        a number, up to the fixed text
    *                             free text, the default

A plain sentence has no line break, no list mark and no bracket, so
the speech engine can read it. A request to `/api/chat` gives its own
script in the field `script`. For a language with other marks, the
field `ends` gives the marks that end a sentence. The field `banned`
gives the marks a sentence cannot have. Each is a string of
characters. The query `latin=1` keeps CJK characters out of a plain
sentence.

The option `shift=` says what a full context does. With `none` a
full context is an error. With `exact` the engine
drops the oldest messages and feeds the rest again. With `soft` the
engine drops them from its cache and feeds nothing again. The soft
form is fast and can cost answers.

## Engines

An engine is one file that ends with an `INSTALL` line. The lists at
the top of `Makefile` name the installed backends and engines. To add
an engine, write the file and add its name to the list of its role.
The headers `backend.h`, `asr.h`, `llm.h` and `tts.h` hold the four
roles.

## Check

    ./test

This program needs no model. The program checks the options, the
helpers, each backend against plain loops, and the feed of a prompt
with an engine that only counts.

    ./check.sh

The script needs the 4 models of the voice chat. The script builds,
starts the server, calls every endpoint and compares 21 answers with
`check.expect`. Words after the name go to `talk` as options, so
`./check.sh gemm=cpu` checks the plain C backend.

    ./bench llm=qwen gemm=metal

This program times a chat engine: the feed of a prompt of 512 tokens
and a reply of 128 tokens, in tokens per second. The option
`profile=1` adds the time of each product.

## Endpoints

| route | method | request | answer |
|---|---|---|---|
| `/` | GET | none | the page |
| `/health` | GET | none | `{"ok":true}` |
| `/api/engines` | GET | none | `{"asr": [...], "llm": [...], ...}` |
| `/api/transcribe` | POST | WAV bytes | `{"texts": [...], "text": "..."}` |
| `/api/chat` | POST | `{"text": "...", "history": [...], "script": "..."}` | `{"text": "...", "thought": "...", "dropped": 0}` |
| `/api/speak` | POST | `{"text": "...", "voice": "F1"}` | WAV bytes |

`/api/chat` takes the sampler options, `llm=`, `think=` and `shift=`
as queries. `dropped` is the count of the oldest messages the engine
dropped. Drop the same count from the history. A reply stops when the
client closes the connection.

`/api/transcribe` takes the queries `language=` and `asr=`. `asr=all`
runs each installed engine, and `texts` has one text per engine. An
error is JSON: `{"error": "..."}`.

## Licences

The code of this repo is under the MIT licence, see `LICENSE`.

Two files follow the code of other authors. No file is a copy.

`whistle.c` follows the Python of github.com/cactus-compute/cactus,
which is under the Apache License, Version 2.0:

> Licensed under the Apache License, Version 2.0 (the "License"); you
> may not use this file except in compliance with the License. You may
> obtain a copy of the License at
> http://www.apache.org/licenses/LICENSE-2.0

`falcon.c` follows the Falcon-H1 graph and the Mamba2 layer of
github.com/ggml-org/llama.cpp, which is under the MIT License:

> Copyright (c) 2023-2026 The ggml authors
>
> Permission is hereby granted, free of charge, to any person
> obtaining a copy of this software and associated documentation
> files (the "Software"), to deal in the Software without
> restriction, including without limitation the rights to use, copy,
> modify, merge, publish, distribute, sublicense, and/or sell copies
> of the Software, and to permit persons to whom the Software is
> furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be
> included in all copies or substantial portions of the Software.

The models have their own licences. Each model file is in a repo of
leok7v on huggingface.co. Four of them are copies of the files of the
authors.

| model | the authors | licence |
|---|---|---|
| Whistle | Cactus-Compute/whistle | Apache 2.0 |
| Falcon-H1 Tiny 90M | tiiuae/Falcon-H1-Tiny-Tool-Calling-90M-GGUF | Falcon-LLM |
| Moonshine tiny | moonshine-ai/moonshine-tiny | MIT |
| Qwen3.5 0.8B | JonathanColetti/Qwen3.5-0.8B-Astra-WeightEdit | Apache 2.0 |
| Supertonic 3 | Supertone/supertonic-3 | BigScience Open RAIL-M |

Read each licence before you give a model to another person.
