#!/bin/sh
# Download the models that talk needs, from huggingface.co/leok7v.
#
#     ./download.sh [directory] [-y] [-a]
#
# With no directory the script asks. After a yes the files go under
# ~/huggingface.co, in the user and repo directories of each model's
# URL. After a no they go under ./models in the same form. The engines
# look in ./models first. -y answers yes to the download of the 4
# models of the voice chat. -a also takes the Qwen chat model and the
# speech separation model.
home="$HOME/huggingface.co"
here=models
url=https://huggingface.co
root=""
yes=no
all=no
for word in "$@"; do
    case "$word" in
        -y) yes=yes ;;
        -a) yes=yes; all=yes ;;
        *) root=$word ;;
    esac
done
ask() {
    if [ "$yes" = yes ]; then
        return 0
    fi
    printf '%s [y/N] ' "$1"
    read -r answer
    case "$answer" in
        y|Y|yes) return 0 ;;
        *) return 1 ;;
    esac
}
missing() {
    total=0
    list=""
    for line in $1; do
        repo=${line%%:*}
        rest=${line#*:}
        file=${rest%%:*}
        size=${rest#*:}
        if [ ! -f "$home/$repo/$file" ] && [ ! -f "$here/$repo/$file" ]; then
            list="$list $line"
            total=$((total + size))
        fi
    done
}
fetch() {
    for line in $list; do
        repo=${line%%:*}
        rest=${line#*:}
        file=${rest%%:*}
        mkdir -p "$root/$repo" || return 1
        echo "download: $repo/$file, about ${rest#*:} MB"
        curl -fL --progress-bar -o "$root/$repo/$file.part" \
            "$url/$repo/resolve/main/$file" &&
            mv "$root/$repo/$file.part" "$root/$repo/$file" || return 1
    done
}
falcon=leok7v/Falcon-H1-Tiny-Tool-Calling-90M-GGUF
voice="leok7v/whistle:whistle.cact:17
$falcon:Falcon-H1-Tiny-90M-Tool-Calling-Q8_0.gguf:116
leok7v/supertonic:supertonic-q8.safetensors:112
leok7v/moonshine-tiny:model.safetensors:108
leok7v/moonshine-tiny:tokenizer.json:2"
qwen="leok7v/Qwen3.5-0.8B-Astra-GGUF:Qwen3.5-0.8B-Astra-Q8_0.gguf:812"
split="leok7v/sr-corrnet-ss:sr-corrnet-ss-2-5spk.safetensors:56"
missing "$voice"
wanted=$list
missing "$qwen"
wanted=$wanted$list
missing "$split"
if [ -z "$root" ] && [ -n "$wanted$list" ]; then
    if ask "download: create the directories under $home?"; then
        root=$home
    else
        root=$here
    fi
fi
missing "$voice"
if [ -z "$list" ]; then
    echo "download: the 4 models of the voice chat are on this machine"
elif ask "download: get about $total MB into $root?"; then
    fetch || exit 1
else
    echo "download: nothing was changed"
    exit 1
fi
missing "$qwen"
if [ -z "$list" ]; then
    echo "download: the Qwen chat model is on this machine"
elif [ "$all" = yes ] || { [ "$yes" = no ] &&
    ask "download: also get the Qwen chat model, about $total MB?"; }; then
    fetch || exit 1
fi
missing "$split"
if [ -z "$list" ]; then
    echo "download: the speech separation model is on this machine"
elif [ "$all" = yes ] || { [ "$yes" = no ] &&
    ask "download: also get the separation model, about $total MB?"; }; then
    fetch || exit 1
fi
