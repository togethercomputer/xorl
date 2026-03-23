import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";

export default defineConfig({
  site: "https://togethercomputer.github.io",
  base: "/xorl",
  integrations: [
    starlight({
      title: "xorl",
      description:
        "High-performance distributed training framework for large language models",
      logo: {
        light: "./src/assets/logo-light.svg",
        dark: "./src/assets/logo-dark.svg",
        replacesTitle: true,
      },
      social: [
        { icon: "github", label: "GitHub", href: "https://github.com/togethercomputer/xorl" },
      ],
      customCss: ["./src/styles/custom.css"],
      sidebar: [
        {
          label: "Getting Started",
          items: [
            { label: "Installation", slug: "getting-started/installation" },
            { label: "Quickstart", slug: "getting-started/quickstart" },
          ],
        },
        {
          label: "Local Training",
          collapsed: true,
          items: [
            { label: "Overview", slug: "training/local_training" },
            { label: "Trainer Architecture", slug: "training/system" },
            { label: "Dataset Loading", slug: "training/dataset_loading" },
            { label: "Checkpointing", slug: "training/checkpointing" },
          ],
        },
        {
          label: "Server Training",
          collapsed: true,
          items: [
            { label: "Overview", slug: "server-training/overview" },
            { label: "Server Architecture", slug: "server-training/architecture" },
            { label: "API Reference", slug: "server-training/api-reference" },
            { label: "RL Training", slug: "server-training/rl-training" },
            { label: "Inference: xorl-sglang", slug: "server-training/sglang" },
            {
              label: "Weight Sync",
              collapsed: true,
              items: [
                { label: "Overview", slug: "server-training/weight-sync/overview" },
                { label: "Backend: nccl_broadcast", slug: "server-training/weight-sync/nccl-broadcast" },
              ],
            },
          ],
        },
        {
          label: "Loss Functions",
          collapsed: true,
          items: [
            { label: "Overview", slug: "loss-functions" },
            { label: "Gradient Accumulation", slug: "loss-functions/gradient-accumulation" },
          ],
        },
        { label: "Supported Models", slug: "models" },
        {
          label: "Parallelism",
          collapsed: true,
          items: [
            { label: "Overview", slug: "parallelism/overview" },
            { label: "Data Parallelism", slug: "parallelism/data_parallelism" },
            { label: "Tensor Parallelism", slug: "parallelism/tensor_parallelism" },
            { label: "Pipeline Parallelism", slug: "parallelism/pipeline_parallelism" },
            { label: "Context Parallelism", slug: "parallelism/context_parallelism" },
            { label: "Expert Parallelism", slug: "parallelism/expert_parallelism" },
          ],
        },
        {
          label: "Adapters",
          collapsed: true,
          items: [
            { label: "LoRA", slug: "adapters/lora" },
            { label: "QLoRA", slug: "adapters/qlora" },
          ],
        },
        {
          label: "MoE",
          collapsed: true,
          items: [
            { label: "Overview", slug: "moe/overview" },
            { label: "Router", slug: "moe/router" },
            { label: "Expert Kernels", slug: "moe/kernels" },
            { label: "Expert Parallelism", slug: "moe/expert-parallelism" },
            { label: "LoRA & QLoRA", slug: "moe/lora" },
            { label: "DeepEP", slug: "moe/deepep" },
          ],
        },
        {
          label: "Config Reference",
          collapsed: true,
          items: [
            { label: "Local Training", slug: "config-reference/local" },
            { label: "Server Training", slug: "config-reference/server" },
          ],
        },
        {
          label: "Contributing",
          collapsed: true,
          items: [
            { label: "Development Guide", slug: "development" },
            { label: "Contributing to Docs", slug: "contributing" },
          ],
        },
        {
          label: "Tests",
          collapsed: true,
          items: [
            { label: "Overview", slug: "testing/overview" },
            { label: "Existing Tests", slug: "testing/existing-tests" },
            { label: "Running Tests", slug: "testing/running-tests" },
            { label: "Adding New Tests", slug: "testing/adding-tests" },
          ],
        },
      ],
    }),
  ],
});
