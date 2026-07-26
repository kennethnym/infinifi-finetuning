#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: eval/pull_metrics.sh [-p PORT] USER@HOST REMOTE_PROJECT_DIR [LOCAL_PROJECT_DIR]

Pull every runs/<run-name>/metrics.json from a remote checkout and recreate
the same layout locally. LOCAL_PROJECT_DIR defaults to the current directory.

SSH authentication uses ~/.ssh/pem/prime-intellect.pem.
EOF
}

identity_file="$HOME/.ssh/pem/prime-intellect.pem"
port=
while getopts ":p:h" option; do
    case "$option" in
        p)
            port=$OPTARG
            ;;
        h)
            usage
            exit 0
            ;;
        :)
            echo "error: -$OPTARG requires a value" >&2
            usage >&2
            exit 2
            ;;
        \?)
            echo "error: unknown option -$OPTARG" >&2
            usage >&2
            exit 2
            ;;
    esac
done
shift $((OPTIND - 1))

if [[ $# -lt 2 || $# -gt 3 ]]; then
    usage >&2
    exit 2
fi

remote=$1
remote_project_dir=${2%/}
local_project_dir=${3:-.}
local_project_dir=${local_project_dir%/}

if [[ -z "$remote" || "$remote" == -* ]]; then
    echo "error: USER@HOST must be a valid SSH destination" >&2
    exit 2
fi
if [[ -z "$remote_project_dir" ]]; then
    remote_project_dir=/
fi
if [[ -z "$local_project_dir" ]]; then
    local_project_dir=/
fi
if [[ -n "$port" && ( ! "$port" =~ ^[0-9]+$ || "$port" -lt 1 || "$port" -gt 65535 ) ]]; then
    echo "error: PORT must be an integer from 1 to 65535" >&2
    exit 2
fi
if [[ ! -r "$identity_file" ]]; then
    echo "error: SSH identity file is not readable: $identity_file" >&2
    exit 2
fi

ssh_options=(-i "$identity_file" -o IdentitiesOnly=yes)
scp_options=(-i "$identity_file" -o IdentitiesOnly=yes)
if [[ -n "$port" ]]; then
    ssh_options+=(-p "$port")
    scp_options+=(-P "$port")
fi

if [[ "$remote_project_dir" == / ]]; then
    remote_runs_dir=/runs
else
    remote_runs_dir="$remote_project_dir/runs"
fi
printf -v quoted_remote_runs_dir '%q' "$remote_runs_dir"

metrics_list=$(mktemp)
trap 'rm -f "$metrics_list"' EXIT

if ! ssh "${ssh_options[@]}" -- "$remote" \
    "find $quoted_remote_runs_dir -mindepth 2 -maxdepth 2 -type f -name metrics.json -print0" \
    >"$metrics_list"; then
    echo "error: could not list metrics below $remote:$remote_runs_dir" >&2
    exit 1
fi

copied=0
while IFS= read -r -d '' remote_metrics; do
    relative_path=${remote_metrics#"$remote_runs_dir/"}
    if [[ "$relative_path" == "$remote_metrics" ]]; then
        echo "error: unexpected remote path: $remote_metrics" >&2
        exit 1
    fi

    if [[ "$local_project_dir" == / ]]; then
        local_metrics="/runs/$relative_path"
    else
        local_metrics="$local_project_dir/runs/$relative_path"
    fi
    mkdir -p -- "$(dirname -- "$local_metrics")"
    scp "${scp_options[@]}" -- "$remote:$remote_metrics" "$local_metrics"
    copied=$((copied + 1))
done <"$metrics_list"

if [[ "$copied" -eq 0 ]]; then
    echo "No metrics.json files found below $remote:$remote_runs_dir" >&2
    exit 1
fi

echo "Copied $copied metrics.json file(s) into $local_project_dir/runs/"
