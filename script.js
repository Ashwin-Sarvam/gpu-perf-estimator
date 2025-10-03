document.addEventListener('DOMContentLoaded', () => {

    // --- Configuration & Data ---
    const HARDWARE_SPECS = {
        "H100 SXM": { vram_gb: 80, mem_bandwidth_tbs: 3.35, fp16_tflops: 1979, fp8_tflops: 3958, fp4_tflops: 0, cost_per_hr_median: 2.49 },
        "H200 SXM": { vram_gb: 141, mem_bandwidth_tbs: 4.8, fp16_tflops: 1979, fp8_tflops: 3958, fp4_tflops: 0, cost_per_hr_median: 3.80 },
        "B200 SXM": { vram_gb: 180, mem_bandwidth_tbs: 8.0, fp16_tflops: 4500, fp8_tflops: 9000, fp4_tflops: 18000, cost_per_hr_median: 5.99 }
    };

    const VRAM_OVERHEAD_FRACTION = 0.05;
    const ETA_COMPUTE = 0.60;
    const ETA_BW = 0.70;
    const GPUS_PER_NODE = 8;

    // --- DOM Elements ---
    const hardwareSelect = document.getElementById('hardware');
    const precisionSelect = document.getElementById('precision');
    const gpuCostInput = document.getElementById('gpuCost');
    const tensorParallelInput = document.getElementById('tensorParallel');
    const calculateBtn = document.getElementById('calculateBtn');
    const manualToggle = document.getElementById('manualParamsToggle');
    const manualSection = document.getElementById('manualParamsSection');
    const loader = document.getElementById('loader');
    const resultsContent = document.getElementById('resultsContent');
    const errorDiv = document.getElementById('error');

    // --- Event Listeners ---
    hardwareSelect.addEventListener('change', () => {
        const selectedHardware = hardwareSelect.value;
        gpuCostInput.value = HARDWARE_SPECS[selectedHardware].cost_per_hr_median;
        
        // Update precision options
        precisionSelect.innerHTML = '<option value="FP16">FP16</option><option value="FP8">FP8</option>';
        if (HARDWARE_SPECS[selectedHardware].fp4_tflops > 0) {
            precisionSelect.innerHTML += '<option value="FP4">FP4</option>';
        }
    });
    
    manualToggle.addEventListener('change', () => {
        manualSection.classList.toggle('hidden', !manualToggle.checked);
    });

    calculateBtn.addEventListener('click', runCalculations);

    // --- Main Calculation Function ---
    async function runCalculations() {
        loader.classList.remove('hidden');
        resultsContent.classList.add('hidden');
        errorDiv.classList.add('hidden');

        try {
            // 1. Get User Inputs
            const inputs = {
                modelId: document.getElementById('modelId').value,
                inputLen: parseInt(document.getElementById('inputLen').value),
                outputLen: parseInt(document.getElementById('outputLen').value),
                hardware: hardwareSelect.value,
                precision: precisionSelect.value,
                numNodes: parseInt(document.getElementById('numNodes').value),
                tensorParallel: parseInt(tensorParallelInput.value),
                gpuCostHr: parseFloat(gpuCostInput.value),
                isManual: manualToggle.checked,
                manualParamsB: parseFloat(document.getElementById('totalParamsManual').value)
            };
            const totalSeqLen = inputs.inputLen + inputs.outputLen;

            // 2. Fetch and prepare model parameters
            let modelParams = {};
            
            // Always fetch config for architecture details, but always use manual input for total_params
                const config = await getHfConfig(inputs.modelId);
            console.log("Fetched config:", config);
                modelParams = getModelParams(config);
            console.log("Parsed model params (before manual total_params):", modelParams);
            
            // Always use manual input for total parameters (more reliable than calculation)
            modelParams.total_params = inputs.manualParamsB * 1e9;
            console.log("Final model params (with manual total_params):", modelParams);

            if (!modelParams.total_params || !modelParams.n_layers) {
                console.error("Missing required parameters:", {
                    total_params: modelParams.total_params,
                    n_layers: modelParams.n_layers,
                    all_params: modelParams
                });
                throw new Error("Could not find total parameters or layer count. Check model ID or enter parameters manually.");
            }
            
            // Display model parameters in a user-friendly format
            displayModelParameters(modelParams);

            // 3. Perform Calculations
            const gpuSpec = HARDWARE_SPECS[inputs.hardware];
            
            const { max_batch_size_per_replica, model_mem_gb, kv_cache_mem_gb_per_replica, total_replicas } = calculateMemoryAndBatchSize(gpuSpec, modelParams, inputs.precision, totalSeqLen, inputs.tensorParallel, inputs.numNodes);
            const prefill_latency_ms = calculatePrefillLatency(gpuSpec, modelParams, inputs.precision, max_batch_size_per_replica, inputs.inputLen, inputs.tensorParallel) * 1000;
            const { throughput_per_replica, throughput_per_request } = calculateDecodeThroughput(gpuSpec, modelParams, inputs.precision, max_batch_size_per_replica, inputs.tensorParallel, totalSeqLen);
            const system_throughput = throughput_per_replica * total_replicas;
            const { input_cost, output_cost } = calculateCosts(prefill_latency_ms / 1000, throughput_per_replica, inputs.gpuCostHr, max_batch_size_per_replica, inputs.inputLen, inputs.numNodes);

            // 4. Display Results
            document.getElementById('res-batchSize').textContent = max_batch_size_per_replica.toLocaleString();
            document.getElementById('res-modelMem').textContent = `${model_mem_gb.toFixed(2)} GB`;
            document.getElementById('res-kvMem').textContent = `${kv_cache_mem_gb_per_replica.toFixed(2)} GB`;
            document.getElementById('res-tensorParallel').textContent = `${inputs.tensorParallel} GPUs (${total_replicas} total replicas)`;
            document.getElementById('res-latency').textContent = `${prefill_latency_ms.toFixed(2)} ms`;
            document.getElementById('res-nodeTput').textContent = `${Math.round(throughput_per_replica).toLocaleString()} tok/s`;
            document.getElementById('res-requestTput').textContent = `${Math.round(throughput_per_request).toLocaleString()} tok/s`;
            document.getElementById('res-sysTput').textContent = `${Math.round(system_throughput).toLocaleString()} tok/s`;
            document.getElementById('res-inputCost').textContent = `$${input_cost.toFixed(4)}`;
            document.getElementById('res-outputCost').textContent = `$${output_cost.toFixed(4)}`;

            resultsContent.classList.remove('hidden');
        } catch (error) {
            errorDiv.textContent = `Error: ${error.message}`;
            errorDiv.classList.remove('hidden');
        } finally {
            loader.classList.add('hidden');
        }
    }

    // --- Helper & Calculation Functions ---
    async function getHfConfig(modelId) {
        // First try to get model info from the official API (includes parameter count)
        const modelInfoUrl = `https://huggingface.co/api/models/${modelId}`;
        console.log("Fetching model info from:", modelInfoUrl);
        
        try {
            const modelResponse = await fetch(modelInfoUrl, {
                headers: {
                    'Accept': 'application/json',
                }
            });
            
            if (modelResponse.ok) {
                const modelInfo = await modelResponse.json();
                console.log("Model info from API:", modelInfo);
                
                // Check if the API response has parameter count
                if (modelInfo.safetensors && modelInfo.safetensors.total_size) {
                    // This gives us file size, not parameter count, but we can use it as a fallback
                    console.log("Found safetensors info:", modelInfo.safetensors);
                }
            }
        } catch (error) {
            console.log("API model info fetch failed, falling back to config:", error);
        }
        
        // Always fetch the config.json for architecture details
        const configUrl = `https://huggingface.co/${modelId}/raw/main/config.json`;
        console.log("Fetching config from:", configUrl);
        
        const response = await fetch(configUrl, {
            headers: {
                'Accept': 'application/json',
            }
        });
        
        if (!response.ok) {
            console.error(`Failed to fetch config. Status: ${response.status}, StatusText: ${response.statusText}`);
            throw new Error(`Could not fetch config from ${configUrl}. Status: ${response.status}. Please try using manual parameter entry or check if the model exists.`);
        }
        
        const config = await response.json();
        console.log("Successfully fetched config:", config);
        return config;
    }
    
    function getModelParams(config) {
        console.log("Available config keys:", Object.keys(config));
        
        // Extract architecture parameters from config (total_params will be set manually)
        const result = {
            total_params: null, // Will be set manually from user input
            n_layers: config.num_hidden_layers || config.n_layers,
            hidden_size: config.hidden_size,
            n_heads: config.num_attention_heads || config.n_heads,
            n_kv_heads: config.num_key_value_heads || config.num_attention_heads || config.n_heads,
            // MoE specific parameters
            num_experts: config.num_experts,
            n_routed_experts: config.n_routed_experts,
            num_experts_per_tok: config.num_experts_per_tok,
            moe_intermediate_size: config.moe_intermediate_size,
            // Additional useful parameters
            vocab_size: config.vocab_size,
            intermediate_size: config.intermediate_size,
            max_position_embeddings: config.max_position_embeddings,
            // Model type and other info
            model_type: config.model_type,
            architectures: config.architectures
        };
        
        console.log("Extracted model parameters:", result);
        return result;
    }
    
    
    function displayModelParameters(modelParams) {
        const summaryDiv = document.getElementById('res-modelParamsSummary');
        const fullDiv = document.getElementById('res-modelParamsFull');
        
        // Create a user-friendly summary
        let summaryHTML = '<div class="model-params-grid">';
        summaryHTML += `<div class="param-item"><strong>Total Parameters:</strong> ${(modelParams.total_params / 1e9).toFixed(1)}B</div>`;
        summaryHTML += `<div class="param-item"><strong>Model Type:</strong> ${modelParams.model_type || 'N/A'}</div>`;
        summaryHTML += `<div class="param-item"><strong>Architecture:</strong> ${modelParams.architectures?.[0] || 'N/A'}</div>`;
        summaryHTML += `<div class="param-item"><strong>Layers:</strong> ${modelParams.n_layers}</div>`;
        summaryHTML += `<div class="param-item"><strong>Hidden Size:</strong> ${modelParams.hidden_size?.toLocaleString() || 'N/A'}</div>`;
        summaryHTML += `<div class="param-item"><strong>Attention Heads:</strong> ${modelParams.n_heads || 'N/A'}</div>`;
        summaryHTML += `<div class="param-item"><strong>KV Heads:</strong> ${modelParams.n_kv_heads || 'N/A'}</div>`;
        summaryHTML += `<div class="param-item"><strong>Vocab Size:</strong> ${modelParams.vocab_size?.toLocaleString() || 'N/A'}</div>`;
        
        // Add MoE specific parameters if present
        if (modelParams.n_routed_experts) {
            summaryHTML += `<div class="param-item"><strong>Total Experts:</strong> ${modelParams.n_routed_experts}</div>`;
            summaryHTML += `<div class="param-item"><strong>Active Experts per Token:</strong> ${modelParams.num_experts_per_tok || 'N/A'}</div>`;
            summaryHTML += `<div class="param-item"><strong>MoE Intermediate Size:</strong> ${modelParams.moe_intermediate_size?.toLocaleString() || 'N/A'}</div>`;
        }
        
        if (modelParams.intermediate_size) {
            summaryHTML += `<div class="param-item"><strong>Intermediate Size:</strong> ${modelParams.intermediate_size.toLocaleString()}</div>`;
        }
        
        if (modelParams.max_position_embeddings) {
            summaryHTML += `<div class="param-item"><strong>Max Position Embeddings:</strong> ${modelParams.max_position_embeddings.toLocaleString()}</div>`;
        }
        
        summaryHTML += '</div>';
        
        summaryDiv.innerHTML = summaryHTML;
        fullDiv.textContent = JSON.stringify(modelParams, null, 2);
    }

    function calculateMemoryAndBatchSize(gpuSpec, modelParams, precision, seqLen, tensorParallel, numNodes) {
        const bytes_per_param = { "FP16": 2, "FP8": 1, "FP4": 1 }[precision];
        const bytes_per_value_cache = 2; // FP16

        // Calculate total model memory
        const total_model_mem_gb = (modelParams.total_params * bytes_per_param) / (1024 ** 3);
        
        // Use user-provided tensor parallelism
        const tensor_parallel_size = tensorParallel;
        
        // Calculate total number of replicas across all nodes
        const total_replicas = Math.floor((numNodes * GPUS_PER_NODE) / tensor_parallel_size);
        
        if (total_replicas === 0) {
            console.log(`Tensor parallelism (${tensor_parallel_size}) exceeds total available GPUs (${numNodes * GPUS_PER_NODE})`);
            return { max_batch_size_per_replica: 0, model_mem_gb: total_model_mem_gb, kv_cache_mem_gb: 0, total_replicas: 0 };
        }
        
        // Model memory per GPU (with tensor parallelism)
        const model_mem_gb = total_model_mem_gb / tensor_parallel_size;
        const available_vram_gb = gpuSpec.vram_gb * (1 - VRAM_OVERHEAD_FRACTION);
        const vram_for_kv = available_vram_gb - model_mem_gb;
        
        // Debug logging
        console.log(`Debug Memory Calculation:`);
        console.log(`- Total model memory: ${total_model_mem_gb.toFixed(2)} GB`);
        console.log(`- Tensor parallel size: ${tensor_parallel_size}`);
        console.log(`- Model memory per GPU: ${model_mem_gb.toFixed(2)} GB`);
        console.log(`- GPU VRAM: ${gpuSpec.vram_gb} GB`);
        console.log(`- Available VRAM: ${available_vram_gb.toFixed(2)} GB`);
        console.log(`- VRAM for KV: ${vram_for_kv.toFixed(2)} GB`);
        
        if (vram_for_kv <= 0) {
            console.log(`Model too large for specified tensor parallelism. Model: ${model_mem_gb.toFixed(1)}GB, Available: ${available_vram_gb.toFixed(1)}GB`);
            return { max_batch_size_per_replica: 0, model_mem_gb: total_model_mem_gb, kv_cache_mem_gb: 0, total_replicas };
        }

        const d_head = modelParams.hidden_size / modelParams.n_heads;
        // KV cache per sequence per GPU (distributed across tensor parallel GPUs)
        const kv_cache_per_seq_per_gpu_gb = (seqLen * modelParams.n_layers * 2 * modelParams.n_kv_heads * d_head * bytes_per_value_cache) / (1024 ** 3) / tensor_parallel_size;
        
        if (kv_cache_per_seq_per_gpu_gb <= 0) return { max_batch_size_per_replica: Infinity, model_mem_gb: total_model_mem_gb, kv_cache_mem_gb: 0, total_replicas };

        // Calculate batch size per replica (based on per-GPU KV cache)
        const max_batch_size_per_replica = Math.floor(vram_for_kv / kv_cache_per_seq_per_gpu_gb);
        // KV cache memory per replica = batch_size * kv_cache_per_seq_per_gpu * tensor_parallel_gpus
        const kv_cache_mem_gb_per_replica = max_batch_size_per_replica * kv_cache_per_seq_per_gpu_gb * tensor_parallel_size;
        
        // Debug logging
        console.log(`Debug KV Cache Calculation:`);
        console.log(`- Sequence length: ${seqLen}`);
        console.log(`- Model layers: ${modelParams.n_layers}`);
        console.log(`- KV heads: ${modelParams.n_kv_heads}`);
        console.log(`- Head dimension: ${d_head}`);
        console.log(`- Tensor parallel size: ${tensor_parallel_size}`);
        console.log(`- KV cache per seq per GPU: ${kv_cache_per_seq_per_gpu_gb.toFixed(3)} GB`);
        console.log(`- KV cache per seq total: ${(kv_cache_per_seq_per_gpu_gb * tensor_parallel_size).toFixed(3)} GB`);
        console.log(`- Available VRAM for KV: ${vram_for_kv.toFixed(2)} GB`);
        console.log(`- Max batch size per replica: ${max_batch_size_per_replica}`);
        
        return { max_batch_size_per_replica, model_mem_gb: total_model_mem_gb, kv_cache_mem_gb_per_replica, total_replicas };
    }

    function calculatePrefillLatency(gpuSpec, modelParams, precision, batchSize, inputSeqLen, tensorParallel) {
        if (batchSize === 0) return 0;
        
        // For MoE models, only use active expert parameters
        // Qwen2.5 72B has 8 experts, 2 active per token
        console.log(`Debug MoE Parameters:`);
        console.log(`- num_experts: ${modelParams.num_experts}`);
        console.log(`- num_experts_per_tok: ${modelParams.num_experts_per_tok}`);
        console.log(`- All model params:`, modelParams);
        
        // Use num_experts_per_tok for active experts, num_experts for total experts
        const active_experts = modelParams.num_experts_per_tok || 2;
        const total_experts = modelParams.num_experts || 8; // This should be 128 from config
        const active_expert_ratio = active_experts / total_experts;
        const active_params = modelParams.total_params * active_expert_ratio;
        
        const total_flops = 2 * batchSize * inputSeqLen * active_params;
        const peak_tflops = gpuSpec[`${precision.toLowerCase()}_tflops`];
        
        // Debug logging
        console.log(`Debug Prefill Latency Calculation:`);
        console.log(`- Precision: ${precision}`);
        console.log(`- Peak TFLOPS: ${peak_tflops}`);
        console.log(`- Batch size: ${batchSize}`);
        console.log(`- Input seq len: ${inputSeqLen}`);
        console.log(`- Total params: ${modelParams.total_params}`);
        console.log(`- Active experts: ${active_experts}`);
        console.log(`- Total experts: ${total_experts}`);
        console.log(`- Active expert ratio: ${active_expert_ratio}`);
        console.log(`- Active params: ${active_params.toExponential(2)}`);
        console.log(`- Total FLOPS: ${total_flops.toExponential(2)}`);
        
        // Use tensor parallel GPUs for prefill (compute-bound phase)
        const effective_tflops = peak_tflops * ETA_COMPUTE * tensorParallel;
        console.log(`- Effective TFLOPS: ${effective_tflops}`);
        
        if (effective_tflops === 0) return Infinity;
        const latency_s = total_flops / (effective_tflops * 1e12);
        console.log(`- Latency: ${latency_s.toFixed(2)} seconds`);
        return latency_s;
    }
    
    function calculateDecodeThroughput(gpuSpec, modelParams, precision, batchSizePerReplica, tensorParallel, seqLen) {
        if (batchSizePerReplica === 0) return { throughput_per_replica: 0, throughput_per_request: 0 };
        
        // Roofline model for decode throughput (memory-bound phase)
        const bytes_per_param = { "FP16": 2, "FP8": 1, "FP4": 1 }[precision];
        const bytes_per_value_cache = 2; // KV cache in FP16
        
        // Calculate memory traffic per decode step per replica
        // 1. Model weights (activated per step) - distributed across tensor parallel GPUs
        // For MoE models, only use active expert parameters
        const active_experts = modelParams.num_experts_per_tok || 2;
        const total_experts = modelParams.num_experts || 8;
        const active_expert_ratio = active_experts / total_experts;
        const active_params = modelParams.total_params * active_expert_ratio;
        const model_weights_bytes = (active_params * bytes_per_param) / tensorParallel;
        
        // 2. KV cache per request (using actual sequence length)
        const d_head = modelParams.hidden_size / modelParams.n_heads;
        const kv_cache_per_request_bytes = seqLen * modelParams.n_layers * 2 * modelParams.n_kv_heads * d_head * bytes_per_value_cache;
        
        // Total memory traffic per decode step per replica
        const total_traffic_per_step_per_replica = model_weights_bytes + (batchSizePerReplica * kv_cache_per_request_bytes);
        
        // Aggregate bandwidth for the tensor parallel GPUs per replica
        const bandwidth_per_replica = gpuSpec.mem_bandwidth_tbs * 1e12 * ETA_BW * tensorParallel;
        
        // Time per decode step per replica (memory-bound)
        const time_per_step_per_replica = total_traffic_per_step_per_replica / bandwidth_per_replica;
        
        // Throughput per replica = batch_size_per_replica / time_per_step_per_replica
        const throughput_per_replica = batchSizePerReplica / time_per_step_per_replica;
        
        // Throughput per request = throughput_per_replica / batch_size_per_replica
        const throughput_per_request = throughput_per_replica / batchSizePerReplica;
        
        return { throughput_per_replica, throughput_per_request };
    }

    function calculateCosts(prefillLatencyS, nodeThroughput, gpuCostHr, batchSize, inputSeqLen, numNodes) {
        if (batchSize === 0) return { input_cost: 0, output_cost: 0 };
        const cost_per_second_node = (gpuCostHr * GPUS_PER_NODE) / 3600;
        
        const cost_per_prefill_batch = prefillLatencyS * cost_per_second_node;
        const total_input_tokens_in_batch = batchSize * inputSeqLen;
        const input_cost = total_input_tokens_in_batch > 0 ? (cost_per_prefill_batch / total_input_tokens_in_batch) * 1_000_000 : 0;
        
        const system_throughput = nodeThroughput * numNodes;
        const time_for_one_million_tokens = system_throughput > 0 ? 1_000_000 / system_throughput : Infinity;
        const output_cost = time_for_one_million_tokens * (cost_per_second_node * numNodes);
        
        return { input_cost, output_cost };
    }
});