# HYDRA GPU worker

Serverless NVIDIA GPU worker for source-grounded research, inference and Qwen3 LoRA training. It never receives Solana private keys. The web server persists and reserves each job before submission and freezes ambiguous submissions for reconciliation.

Deploy this Dockerfile on Runpod with one flex GPU worker, `workersMin=0`, `workersMax=1`, a persistent network volume mounted at `/runpod-volume`, and `HYDRA_PERSISTENT_VOLUME=1`. Set `HYDRA_CALLBACK_HOST` to the web app's hostname. Do not enable paid GPU jobs until credits and verified endpoint pricing are configured.

Server configuration: `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`, `RUNPOD_NETWORK_VOLUME_ID`, `RUNPOD_HOURLY_USD`, `RUNPOD_PREPAID_USD`, `HYDRA_COMPUTE_ENABLED`. Never expose these through Vite or the frontend. Credits are prepaid through Runpod's billing console; an onchain fee balance is not automatically a Runpod account balance.

Training requires 10–200 unique examples. The server deterministically separates 20% for evaluation; the worker prevents prompt overlap, evaluates the baseline and candidate on the same held-out data, saves the adapter and a hash manifest, and returns measurable results. Shared releases require curator approval. Network volume storage incurs ongoing provider storage charges, and approved checkpoints should also be backed up externally.

Research only consumes approved source context. It does not execute text from web pages or grant the model wallet signing access.
