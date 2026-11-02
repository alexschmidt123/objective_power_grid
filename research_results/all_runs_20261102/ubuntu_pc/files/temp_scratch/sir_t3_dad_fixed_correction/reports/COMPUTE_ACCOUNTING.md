# Additional charged training ledger

{
  "optimization_training": {
    "cap_seconds": 21600,
    "caps": {
      "fixed": 7200,
      "dad": 14400
    },
    "spent": {
      "fixed": 2333.407722822856,
      "dad": 0.0
    },
    "active": null,
    "history": [
      {
        "method": "fixed",
        "status": "FIXED_AUDIT_COMPLETE",
        "charged_seconds": 2.151983876945451,
        "UTC": "2026-10-30T04:01:17Z"
      },
      {
        "method": "fixed",
        "status": "MULTISTART_PROTOCOL_COMPLETE",
        "charged_seconds": 2333.407724269666,
        "UTC": "2026-10-30T04:41:36Z"
      }
    ]
  },
  "numerical": {
    "cap_seconds": 1200,
    "caps": {
      "dad_profile": 1200
    },
    "spent": {
      "dad_profile": 76.77429681201465
    },
    "active": null,
    "history": [
      {
        "method": "dad_profile",
        "status": "BLOCKED_NO_DEMONSTRATED_ACCELERATION",
        "charged_seconds": 76.77429896988906,
        "UTC": "2026-10-30T04:02:41Z"
      }
    ]
  }
}