#!/bin/sh
# The end to end gate: build, start the server, call every endpoint,
# compare the answers with check.expect. Exit 0 when they are the same.
# Words after the name go to talk as options, for example gemm=cpu.
cd "$(dirname "$0")" || exit 1
make >/dev/null || exit 1
mkdir -p tmp/check
out=tmp/check/out.txt
url=http://127.0.0.1:8081
./talk temperature=0 dry=0 "$@" >tmp/check/talk.log 2>&1 &
pid=$!
trap 'kill $pid 2>/dev/null; wait $pid 2>/dev/null' EXIT
curl -s -o /dev/null --retry 60 --retry-connrefused --retry-delay 1 $url/health
call() {
    name=$1
    shift
    printf '%s: ' "$name"
    curl -s -m 300 -w ' [%{http_code}]\n' "$@"
}
sum() {
    name=$1
    shift
    file=tmp/check/$name.bin
    code=$(curl -s -m 300 -o $file -w '%{http_code}' "$@")
    hash=$(md5 -q $file 2>/dev/null || md5sum <$file | cut -d ' ' -f 1)
    echo "$name: $hash [$code]"
}
wave() {
    name=$1
    shift
    file=tmp/check/$name.bin
    code=$(curl -s -m 300 -o $file -w '%{http_code}' "$@")
    form="$(head -c 4 $file) $(head -c 12 $file | tail -c 4)"
    echo "$name: $form, $((($(wc -c <$file) - 44) / 88200)) s [$code]"
}
timeless() {
    sed -E 's/,"seconds":[0-9.]+//g'
}
head -c 11000000 /dev/zero >tmp/check/big.bin
{
    call health $url/health
    sum page $url/
    call missing $url/nope
    call method -X PUT $url/health
    call speak-empty -X POST -d '{}' $url/api/speak
    call speak-voice -X POST -d '{"text":"hi","voice":"X9"}' $url/api/speak
    call speak-blank -X POST -d '{"text":" "}' $url/api/speak
    wave speak -X POST -d '{"text":"Hello world.","voice":"M1"}' $url/api/speak
    call wav-bad -X POST --data-binary 'not a wav' $url/api/transcribe
    call engines $url/api/engines
    call wav -X POST --data-binary @tmp/check/speak.bin $url/api/transcribe |
        timeless
    call wav-all -X POST --data-binary @tmp/check/speak.bin \
        "$url/api/transcribe?language=en&asr=all" | timeless
    call wav-engine -X POST --data-binary @tmp/check/speak.bin \
        "$url/api/transcribe?asr=nobody"
    call chat-empty -X POST -d '{}' $url/api/chat
    call chat -X POST -d '{"text":"What is the capital of France?"}' \
        $url/api/chat
    call chat-history -X POST -d '{"text":"And of Italy?","history":[
        {"role":"user","content":"Café 😀 \"q\"\n: What is
        the capital of France?"},
        {"role":"assistant","content":"The capital of France is Paris."}]}' \
        $url/api/chat
    call chat-script -X POST -d '{"text":"Tell me about the sea.",
        "script":"S"}' $url/api/chat
    call chat-engine -X POST -d '{"text":"hi"}' "$url/api/chat?llm=nobody"
    call post-missing -X POST -d '{}' $url/api/nope
    call too-large -X POST --data-binary @tmp/check/big.bin $url/api/chat
    call after $url/health
} >$out 2>&1
if diff -u check.expect $out; then
    echo "check: PASS, $(wc -l <$out | tr -d ' ') answers"
else
    echo "check: FAIL"
    exit 1
fi
