"""Discord notifications and chart generation."""

from .discord import (
    send_discord_alert,
    send_entry_alert,
    send_exit_alert,
    send_status_update,
    send_error_alert,
    send_startup_notification,
    send_regime_change_alert,
    send_regime_status_update,
    DiscordConfig
)

from .charts import (
    generate_velocity_chart,
    generate_signal_chart
)

__all__ = [
    'send_discord_alert',
    'send_entry_alert',
    'send_exit_alert',
    'send_status_update',
    'send_error_alert',
    'send_startup_notification',
    'send_regime_change_alert',
    'send_regime_status_update',
    'DiscordConfig',
    'generate_velocity_chart',
    'generate_signal_chart'
]
