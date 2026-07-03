import importlib.util
from pathlib import Path

import pytest


pytestmark = pytest.mark.cpu


def _load_module():
    path = Path("experiments/local_benchmark/scripts/glm_5_1_make_mixed_pool_manifest.py")
    spec = importlib.util.spec_from_file_location("glm5_mixed_pool_manifest", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _node(name, pool, alloc=8, ready=True, unschedulable=False, ready_since=None, taints=None):
    ready_condition = {"type": "Ready", "status": "True" if ready else "False"}
    if ready_since is not None:
        ready_condition["lastTransitionTime"] = ready_since
    return {
        "metadata": {"name": name, "labels": {"node-group": pool}},
        "spec": {"unschedulable": unschedulable, "taints": taints or []},
        "status": {
            "allocatable": {"nvidia.com/gpu": str(alloc)},
            "conditions": [ready_condition],
        },
    }


def _pod(node_name, gpus, phase="Running"):
    return {
        "status": {"phase": phase},
        "spec": {
            "nodeName": node_name,
            "containers": [{"resources": {"limits": {"nvidia.com/gpu": str(gpus)}}}],
        },
    }


def _storage_pod(node_name, ready=True, phase="Running"):
    return {
        "metadata": {"labels": {"app": "weka-storage-node"}},
        "status": {
            "phase": phase,
            "containerStatuses": [{"ready": ready}, {"ready": True}, {"ready": True}],
        },
        "spec": {"nodeName": node_name},
    }


def test_available_slots_count_default_and_nccl_pools():
    module = _load_module()
    nodes_doc = {
        "items": [
            _node("default-a", "default"),
            _node("default-b", "default"),
            _node("nccl-a", "nccl"),
            _node("nccl-b", "nccl", ready=False),
            _node("nccl-c", "nccl", unschedulable=True),
            _node("other-a", "other"),
        ]
    }
    pods_doc = {"items": [_pod("default-a", 4), _pod("nccl-a", 8, phase="Succeeded")]}

    slots = module._available_slots_by_pool(
        nodes_doc,
        pods_doc,
        gpus_per_pod=4,
        excluded_by_pool={"default": {"default-b"}},
    )

    assert slots == {"default": 1, "nccl": 2}


def test_available_slots_skip_unhealthy_storage_nodes_by_default():
    module = _load_module()
    nodes_doc = {
        "items": [
            _node("default-a", "default"),
            _node("default-b", "default"),
            _node("nccl-a", "nccl"),
        ]
    }
    pods_doc = {
        "items": [
            _storage_pod("default-a", ready=False),
            _storage_pod("default-b", ready=True),
            _storage_pod("nccl-a", phase="Pending"),
        ]
    }

    slots = module._available_slots_by_pool(nodes_doc, pods_doc, gpus_per_pod=4)

    assert slots == {"default": 2, "nccl": 0}
    assert module._unhealthy_storage_nodes(pods_doc) == {"default-a", "nccl-a"}


def test_available_slots_can_allow_unhealthy_storage_nodes():
    module = _load_module()
    nodes_doc = {"items": [_node("default-a", "default")]}
    pods_doc = {"items": [_storage_pod("default-a", ready=False)]}

    slots = module._available_slots_by_pool(
        nodes_doc,
        pods_doc,
        gpus_per_pod=4,
        require_storage_ready=False,
    )

    assert slots == {"default": 2, "nccl": 0}


def test_available_slots_skip_recently_recovered_nodes():
    module = _load_module()
    nodes_doc = {
        "items": [
            _node("default-old", "default", ready_since="2026-05-19T15:20:00Z"),
            _node("default-recent", "default", ready_since="2026-05-19T15:29:30Z"),
            _node("nccl-no-transition", "nccl"),
        ]
    }
    pods_doc = {"items": []}

    slots = module._available_slots_by_pool(
        nodes_doc,
        pods_doc,
        gpus_per_pod=8,
        min_ready_age_seconds=120,
        now=module.dt.datetime(2026, 5, 19, 15, 30, tzinfo=module.dt.UTC),
    )

    assert slots == {"default": 1, "nccl": 1}


def test_available_slots_skip_nodes_with_blocking_taints():
    module = _load_module()
    nodes_doc = {
        "items": [
            _node(
                "default-a",
                "default",
                taints=[{"key": "node-group", "value": "default", "effect": "NoSchedule"}],
            ),
            _node(
                "nccl-a",
                "nccl",
                taints=[{"key": "node-group", "value": "nccl", "effect": "NoSchedule"}],
            ),
            _node(
                "nccl-degraded",
                "nccl",
                taints=[
                    {"key": "node-group", "value": "nccl", "effect": "NoSchedule"},
                    {"key": "nvidia.com/gpu-error", "value": "degraded-hardware", "effect": "NoSchedule"},
                ],
            ),
            _node(
                "nccl-unreachable",
                "nccl",
                taints=[
                    {"key": "node-group", "value": "nccl", "effect": "NoSchedule"},
                    {"key": "node.kubernetes.io/unreachable", "effect": "NoExecute"},
                ],
            ),
            _node(
                "nccl-prefer",
                "nccl",
                taints=[
                    {"key": "node-group", "value": "nccl", "effect": "NoSchedule"},
                    {"key": "soft-risk", "effect": "PreferNoSchedule"},
                ],
            ),
        ]
    }
    pods_doc = {"items": []}

    slots = module._available_slots_by_pool(nodes_doc, pods_doc, gpus_per_pod=8)

    assert slots == {"default": 1, "nccl": 2}


def test_configure_job_sets_matching_pool_and_rank_env():
    module = _load_module()
    base = {
        "metadata": {"name": "base", "labels": {}},
        "spec": {
            "template": {
                "metadata": {"labels": {"app": "base"}},
                "spec": {
                    "nodeSelector": {"node-group": "default"},
                    "tolerations": [{"key": "node-group", "operator": "Equal", "value": "default"}],
                    "containers": [{"env": []}],
                },
            }
        },
    }

    job = module._configure_job(
        base,
        name="mixed-nccl",
        variant="mixed",
        node_group="nccl",
        completions=3,
        rank_offset=5,
        nnodes=8,
        master_port=29719,
        rdzv_group_id="group",
        rdzv_dir="/tmp/rdzv",
        result_root="/tmp/results",
    )

    spec = job["spec"]
    pod_spec = spec["template"]["spec"]
    env = {item["name"]: item["value"] for item in pod_spec["containers"][0]["env"]}

    assert spec["completions"] == 3
    assert spec["parallelism"] == 3
    assert pod_spec["nodeSelector"]["node-group"] == "nccl"
    assert pod_spec["tolerations"][0]["value"] == "nccl"
    assert env["NODE_RANK_OFFSET"] == "5"
    assert env["NNODES"] == "8"
    assert env["RDZV_DIR_OVERRIDE"] == "/tmp/rdzv"
    assert env["NCCL_RAS_ENABLE"] == "0"
    assert env["TORCH_NCCL_TRACE_BUFFER_SIZE"] == "1048576"
    assert env["TORCH_NCCL_DUMP_ON_TIMEOUT"] == "1"
    assert env["XORL_DISABLE_HOST_INVENTORY"] == "1"
    assert env["XORL_GLM5_DETERMINISTIC_DKV_ACCUM_DTYPE"] == "input"


def test_configure_job_adds_transient_hostname_exclusions():
    module = _load_module()
    base = {
        "metadata": {"name": "base", "labels": {}},
        "spec": {
            "template": {
                "metadata": {"labels": {"app": "base"}},
                "spec": {
                    "nodeSelector": {"node-group": "default"},
                    "tolerations": [],
                    "affinity": {
                        "nodeAffinity": {
                            "requiredDuringSchedulingIgnoredDuringExecution": {
                                "nodeSelectorTerms": [
                                    {
                                        "matchExpressions": [
                                            {
                                                "key": "kubernetes.io/hostname",
                                                "operator": "NotIn",
                                                "values": ["persistent-bad"],
                                            }
                                        ]
                                    }
                                ]
                            }
                        }
                    },
                    "containers": [{"env": []}],
                },
            }
        },
    }

    job = module._configure_job(
        base,
        name="mixed-default",
        variant="mixed",
        node_group="default",
        completions=1,
        rank_offset=0,
        nnodes=8,
        master_port=29719,
        rdzv_group_id="group",
        rdzv_dir="/tmp/rdzv",
        result_root="/tmp/results",
        excluded_hostnames={"transient-dirty"},
    )

    expressions = job["spec"]["template"]["spec"]["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"][0]["matchExpressions"]
    hostname_expr = next(expr for expr in expressions if expr["key"] == "kubernetes.io/hostname")

    assert hostname_expr["operator"] == "NotIn"
    assert hostname_expr["values"] == ["persistent-bad", "transient-dirty"]


def test_parse_excluded_nodes_supports_pool_specific_and_global():
    module = _load_module()

    excluded = module._parse_excluded_nodes(["both-pools", "default:default-only", "nccl:nccl-only"])

    assert excluded == {
        "default": {"both-pools", "default-only"},
        "nccl": {"both-pools", "nccl-only"},
    }


def test_validate_config_path_rejects_wrong_launcher_config():
    module = _load_module()
    job = {
        "spec": {
            "template": {
                "spec": {
                    "containers": [
                        {
                            "command": [
                                "/bin/bash",
                                "-lc",
                                'CONFIG_PATH="${REPO_ROOT}/experiments/local_benchmark/configs/right.yaml"\n',
                            ]
                        }
                    ]
                }
            }
        }
    }

    actual = module._validate_config_path(
        job,
        Path("manifest.yaml"),
        "experiments/local_benchmark/configs/right.yaml",
    )

    assert actual.endswith("experiments/local_benchmark/configs/right.yaml")
    with pytest.raises(ValueError, match="expected it to contain"):
        module._validate_config_path(
            job,
            Path("manifest.yaml"),
            "experiments/local_benchmark/configs/wrong.yaml",
        )


def test_infer_result_root_from_selected_manifests(monkeypatch):
    module = _load_module()

    def job_with_result_root(result_root):
        return {
            "spec": {
                "template": {
                    "spec": {
                        "containers": [
                            {
                                "command": [
                                    "/bin/bash",
                                    "-lc",
                                    f'RESULT_ROOT="${{RESULT_ROOT_OVERRIDE:-{result_root}}}"\n',
                                ]
                            }
                        ]
                    }
                }
            }
        }

    jobs = {
        Path("default.yaml"): job_with_result_root("/shared/glm-default"),
        Path("nccl.yaml"): job_with_result_root("/shared/glm-default"),
        Path("other.yaml"): job_with_result_root("/shared/glm-other"),
    }
    monkeypatch.setattr(module, "_load_job", lambda path: jobs[Path(path)])

    assert module._infer_result_root_from_manifests([Path("default.yaml"), Path("nccl.yaml")]) == "/shared/glm-default"
    with pytest.raises(ValueError, match="different RESULT_ROOT defaults"):
        module._infer_result_root_from_manifests([Path("default.yaml"), Path("other.yaml")])
