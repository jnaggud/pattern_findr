# Local configuration

Copy `.env.example` to `.env` for the Streamlit dashboard and fill in only the values you need. The application loads that file through python-dotenv. Command-line tools read environment variables; export the values before launching them.

| Variable | Use |
| --- | --- |
| `POLYGON_API_KEY` | Optional Polygon market data |
| `DATABENTO_API_KEY` | Optional Databento market data |
| `DISCORD_WEBHOOK_URL` | Default legacy signal notification endpoint |
| `DISCORD_TEST_WEBHOOK_URL` | Optional test endpoint for the modular traders |
| `DISCORD_DELAYED_WEBHOOK_URL` | Delayed-signal bot |
| `DISCORD_SECONDARY_DELAYED_WEBHOOK_URL` | Optional secondary delayed-signal destination |
| `DISCORD_ANNOUNCEMENTS_WEBHOOK_URL` | Weekly report bot |

Blank values leave these optional integrations unconfigured. The main research workflow can use yfinance without paid credentials. Some saved-bundle workflows read `polygon_api_key` or `discord_webhook` from a local JSON configuration; the committed examples have empty values. Keep any populated versions private and do not commit them.

The legacy secondary-channel maps contain empty endpoints. Configure these locally only if secondary delivery is required. Starting the dashboard does not require running any notification bot or enabling the Sierra Chart bridge.

Generated state, local databases, model weights, and credentials are not portable defaults. Start with your own local state and evaluate a strategy before enabling notifications or any execution integration.
