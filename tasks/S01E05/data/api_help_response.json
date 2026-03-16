{
  "ok": true,
  "action": "help",
  "help": {
    "actions": [
      {
        "action": "help",
        "requires": [],
        "optional": [],
        "about": "Show available actions and parameters."
      },
      {
        "action": "reconfigure",
        "requires": [
          "route"
        ],
        "optional": [],
        "about": "Enable reconfigure mode for the given route."
      },
      {
        "action": "getstatus",
        "requires": [
          "route"
        ],
        "optional": [],
        "about": "Get current status for the given route."
      },
      {
        "action": "setstatus",
        "requires": [
          "route",
          "value"
        ],
        "optional": [],
        "about": "Set route status while in reconfigure mode.",
        "allowed_values": [
          "RTOPEN",
          "RTCLOSE"
        ]
      },
      {
        "action": "save",
        "requires": [
          "route"
        ],
        "optional": [],
        "about": "Exit reconfigure mode for the given route."
      }
    ],
    "route_format": "[a-z]-[0-9]{1,2} (case-insensitive)",
    "status_values": {
      "RTOPEN": "open",
      "RTCLOSE": "close"
    },
    "notes": [
      "To change status of the road, you must first set it to \"reconfigure\" mode."
    ]
  }
}