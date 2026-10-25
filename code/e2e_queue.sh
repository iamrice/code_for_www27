#!/bin/zsh
# usage: e2e_queue.sh JOBFILE   (each line: tag|extra dfl.py args); skips finished runs
cd "${0:A:h}"; export OBJC_DISABLE_INITIALIZE_FORK_SAFETY=YES
while IFS='|' read -r tag args; do
  [[ -z "$tag" || "$tag" == \#* ]] && continue
  out=../results/e2e/$tag.json
  [[ -f $out ]] && continue
  echo "=== $tag $(date +%H:%M:%S)"
  eval python3 -u dfl.py $args --out $out > ../results/e2e/$tag.log 2>&1
  tail -1 ../results/e2e/$tag.log
done < $1
