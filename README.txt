GENERATE -> HOST ORCHESTRATOR BRIDGE
======================================

Files in this patch:

  orchestrator.py
      Replace <toolkit-root>/dev/orchestrator.py

  gui/compose.dev.yaml
      Replace <toolkit-root>/dev/gui/compose.dev.yaml

  gui/src/lib/components/Generate.svelte
      Replace the existing file

  gui/src/lib/server/orchestrator.ts
      New file

  gui/src/routes/api/projects/[projectId]/generate/+server.ts
      New file

Runtime design
--------------
The GUI remains inside Docker. The orchestrator remains a host process in the
active eckf-pitch Conda environment. They communicate only through:

  <toolkit-root>/dev/.runtime/orchestrator.sock

The container sees that same socket as:

  /orchestrator-runtime/orchestrator.sock

Start order after applying the patch
------------------------------------
From the toolkit root, with eckf-pitch active:

  python dev/orchestrator.py --serve

Leave that process running. It will create dev/.runtime itself and wait for
Generate requests.

Because compose.dev.yaml gained a new bind mount, recreate the GUI container
once in another terminal:

  cd dev/gui
  sudo docker compose -f compose.dev.yaml up -d

After that, ordinary stop/start retains the mount:

  sudo docker compose -f compose.dev.yaml stop
  sudo docker compose -f compose.dev.yaml start

But the host orchestrator service must also be running whenever Generate is
used.

Generate behavior
-----------------
1. GUI saves/overwrites/clones the project exactly as before.
2. GUI POSTs the saved project ID to /api/projects/<id>/generate.
3. SvelteKit sends {"project":"<id>"} through the Unix socket.
4. Host orchestrator processes the project under eckf-pitch.
5. The browser request waits for completion and reports success/failure.

If orchestration fails after the save succeeded, the GUI explicitly says that
the project was saved but generation failed. The project is not lost.
