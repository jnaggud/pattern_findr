/*
Pattern FindR Signal Display - ACSIL Custom Study for Sierra Chart

This study fetches trading signals from the Pattern_FindR system via HTTP
and displays entry/exit markers on the chart.

Setup:
1. Copy this file to your Sierra Chart ACS_Source folder
2. Build the custom study DLL (Analysis > Build Custom Studies DLL)
3. Add study to chart (Analysis > Studies > Add Custom Study)
4. Configure the Signal Server URL to point to your Pattern_FindR server
5. Set the Symbol Filter to match your chart symbol (e.g., "ES", "GC", "NQ")
*/

#include "sierrachart.h"
#include <string>
#include <cstring>
#include <cstdlib>

SCDLLName("Pattern FindR Signal Display")

// Persistent data structure
struct PFRPersistent {
    int lastFetchSecond;
    std::string lastSignalId;
    int drawingNumber;
    int requestID;
};

/*
    Simple JSON value extractor.
    Finds a key in JSON and extracts its value.
*/
std::string ExtractJSONValue(const std::string& json, const std::string& key) {
    std::string searchKey = "\"" + key + "\"";
    size_t keyPos = json.find(searchKey);
    if (keyPos == std::string::npos) return "";

    size_t colonPos = json.find(':', keyPos);
    if (colonPos == std::string::npos) return "";

    size_t valueStart = json.find_first_not_of(" \t\n\r", colonPos + 1);
    if (valueStart == std::string::npos) return "";

    std::string value;
    if (json[valueStart] == '"') {
        // String value
        size_t valueEnd = json.find('"', valueStart + 1);
        if (valueEnd != std::string::npos) {
            value = json.substr(valueStart + 1, valueEnd - valueStart - 1);
        }
    } else {
        // Numeric or boolean value
        size_t valueEnd = json.find_first_of(",}\n", valueStart);
        if (valueEnd != std::string::npos) {
            value = json.substr(valueStart, valueEnd - valueStart);
            // Trim whitespace
            size_t lastNonSpace = value.find_last_not_of(" \t\n\r");
            if (lastNonSpace != std::string::npos) {
                value = value.substr(0, lastNonSpace + 1);
            }
        }
    }

    return value;
}

SCSFExport scsf_PFR_SignalDisplay(SCStudyInterfaceRef sc) {
    // Subgraphs for drawing
    SCSubgraphRef BuyArrow = sc.Subgraph[0];
    SCSubgraphRef SellArrow = sc.Subgraph[1];
    SCSubgraphRef PositionLine = sc.Subgraph[2];

    // Inputs
    SCInputRef SignalURL = sc.Input[0];
    SCInputRef SymbolFilter = sc.Input[1];
    SCInputRef PollIntervalSeconds = sc.Input[2];
    SCInputRef ArrowOffsetTicks = sc.Input[3];
    SCInputRef ShowPositionLine = sc.Input[4];
    SCInputRef MaxSignalsToDisplay = sc.Input[5];
    SCInputRef EnableAlertSound = sc.Input[6];

    if (sc.SetDefaults) {
        sc.GraphName = "Pattern FindR Signals";
        sc.StudyDescription = "Displays trading signals from Pattern_FindR system";
        sc.AutoLoop = 0;
        sc.GraphRegion = 0;
        sc.UpdateAlways = 1;

        BuyArrow.Name = "Buy Signal";
        BuyArrow.DrawStyle = DRAWSTYLE_ARROWUP;
        BuyArrow.PrimaryColor = RGB(0, 200, 0);
        BuyArrow.LineWidth = 3;
        BuyArrow.DrawZeros = false;

        SellArrow.Name = "Exit Signal";
        SellArrow.DrawStyle = DRAWSTYLE_ARROWDOWN;
        SellArrow.PrimaryColor = RGB(200, 0, 0);
        SellArrow.LineWidth = 3;
        SellArrow.DrawZeros = false;

        PositionLine.Name = "Entry Price Line";
        PositionLine.DrawStyle = DRAWSTYLE_LINE;
        PositionLine.PrimaryColor = RGB(0, 128, 255);
        PositionLine.LineWidth = 2;
        PositionLine.LineStyle = LINESTYLE_DASH;
        PositionLine.DrawZeros = false;

        SignalURL.Name = "Signal Server URL";
        SignalURL.SetString("https://nonhereditarily-warless-holden.ngrok-free.dev/signals");
        SignalURL.SetDescription("URL of the Pattern_FindR signal server");

        SymbolFilter.Name = "Symbol Filter";
        SymbolFilter.SetString("ES");
        SymbolFilter.SetDescription("Sierra Chart symbol to filter signals (e.g., ES, GC, NQ)");

        PollIntervalSeconds.Name = "Poll Interval (seconds)";
        PollIntervalSeconds.SetInt(2);
        PollIntervalSeconds.SetIntLimits(1, 60);
        PollIntervalSeconds.SetDescription("How often to check for new signals");

        ArrowOffsetTicks.Name = "Arrow Offset (ticks)";
        ArrowOffsetTicks.SetInt(5);
        ArrowOffsetTicks.SetIntLimits(0, 50);
        ArrowOffsetTicks.SetDescription("Distance from price to draw arrows");

        ShowPositionLine.Name = "Show Position Line";
        ShowPositionLine.SetYesNo(true);
        ShowPositionLine.SetDescription("Draw horizontal line at entry price");

        MaxSignalsToDisplay.Name = "Max Signals to Display";
        MaxSignalsToDisplay.SetInt(20);
        MaxSignalsToDisplay.SetIntLimits(1, 100);

        EnableAlertSound.Name = "Enable Alert Sound";
        EnableAlertSound.SetYesNo(true);

        return;
    }

    // Get or initialize persistent data
    PFRPersistent* p_Data = reinterpret_cast<PFRPersistent*>(sc.GetPersistentPointer(1));
    if (p_Data == nullptr) {
        p_Data = new PFRPersistent();
        p_Data->lastFetchSecond = 0;
        p_Data->drawingNumber = 10000;
        p_Data->requestID = 0;
        sc.SetPersistentPointer(1, p_Data);
    }

    // Throttle requests using simple second counter
    int currentSecond = sc.CurrentSystemDateTime.GetTimeInSeconds();
    int secondsSinceLastFetch = currentSecond - p_Data->lastFetchSecond;

    if (secondsSinceLastFetch < PollIntervalSeconds.GetInt() && p_Data->lastFetchSecond != 0) {
        return;
    }
    p_Data->lastFetchSecond = currentSecond;

    // Make HTTP request using sc.MakeHTTPRequest
    SCString url = SignalURL.GetString();

    // Check if we have a pending request
    if (p_Data->requestID != 0) {
        // Check if request is complete
        SCString response;
        if (sc.HTTPRequestID == p_Data->requestID) {
            // Request complete, get response
            response = sc.HTTPResponse;
            p_Data->requestID = 0;
        } else {
            return;  // Still waiting
        }

        if (response.GetLength() == 0) {
            return;
        }

        // Parse response
        std::string jsonResponse(response.GetChars());

        // Clear previous drawings
        sc.DeleteACSChartDrawing(sc.ChartNumber, TOOL_DELETE_ALL, 0);

        float tickSize = sc.TickSize;
        int arrowOffset = ArrowOffsetTicks.GetInt();
        int maxSignals = MaxSignalsToDisplay.GetInt();
        int signalCount = 0;
        SCString symbolFilter = SymbolFilter.GetString();

        // Find signals array
        size_t signalsStart = jsonResponse.find("\"signals\"");
        if (signalsStart == std::string::npos) return;

        size_t arrayStart = jsonResponse.find('[', signalsStart);
        if (arrayStart == std::string::npos) return;

        size_t pos = arrayStart;
        bool foundNewSignal = false;

        while (signalCount < maxSignals) {
            size_t objStart = jsonResponse.find('{', pos);
            if (objStart == std::string::npos) break;

            size_t objEnd = jsonResponse.find('}', objStart);
            if (objEnd == std::string::npos) break;

            std::string signalObj = jsonResponse.substr(objStart, objEnd - objStart + 1);

            std::string sierraSymbol = ExtractJSONValue(signalObj, "sierra_symbol");
            if (!symbolFilter.IsEmpty() && sierraSymbol != std::string(symbolFilter.GetChars())) {
                pos = objEnd + 1;
                continue;
            }

            std::string signalType = ExtractJSONValue(signalObj, "signal_type");
            std::string signalId = ExtractJSONValue(signalObj, "id");
            std::string priceStr = ExtractJSONValue(signalObj, "price");

            if (priceStr.empty()) {
                pos = objEnd + 1;
                continue;
            }

            double price = std::atof(priceStr.c_str());
            int barIndex = sc.ArraySize - 1;

            s_UseTool marker;
            marker.Clear();
            marker.ChartNumber = sc.ChartNumber;
            marker.DrawingType = DRAWING_MARKER;

            if (signalType == "ENTRY") {
                marker.MarkerType = MARKER_ARROWUP;
                marker.Color = RGB(0, 200, 0);
                marker.BeginValue = static_cast<float>(price) - (arrowOffset * tickSize);
            } else if (signalType == "EXIT") {
                marker.MarkerType = MARKER_ARROWDOWN;
                marker.Color = RGB(200, 0, 0);
                marker.BeginValue = static_cast<float>(price) + (arrowOffset * tickSize);
            } else {
                pos = objEnd + 1;
                continue;
            }

            marker.LineWidth = 3;
            marker.BeginDateTime = sc.BaseDateTimeIn[barIndex];
            marker.AddMethod = UTAM_ADD_OR_ADJUST;
            marker.LineNumber = p_Data->drawingNumber++;

            sc.UseTool(marker);

            if (!signalId.empty() && signalId != p_Data->lastSignalId) {
                foundNewSignal = true;
                p_Data->lastSignalId = signalId;
            }

            signalCount++;
            pos = objEnd + 1;
        }

        // Draw position line
        if (ShowPositionLine.GetYesNo()) {
            size_t positionsStart = jsonResponse.find("\"active_positions\"");
            if (positionsStart != std::string::npos) {
                size_t posObjStart = jsonResponse.find('{', positionsStart);
                if (posObjStart != std::string::npos) {
                    size_t posObjEnd = jsonResponse.find('}', posObjStart);
                    if (posObjEnd != std::string::npos) {
                        std::string posObj = jsonResponse.substr(posObjStart, posObjEnd - posObjStart + 1);
                        std::string posSierraSymbol = ExtractJSONValue(posObj, "sierra_symbol");

                        if (symbolFilter.IsEmpty() || posSierraSymbol == std::string(symbolFilter.GetChars())) {
                            std::string entryPriceStr = ExtractJSONValue(posObj, "entry_price");
                            if (!entryPriceStr.empty()) {
                                double entryPrice = std::atof(entryPriceStr.c_str());

                                s_UseTool line;
                                line.Clear();
                                line.ChartNumber = sc.ChartNumber;
                                line.DrawingType = DRAWING_HORIZONTALLINE;
                                line.Color = RGB(0, 128, 255);
                                line.LineWidth = 2;
                                line.LineStyle = LINESTYLE_DASH;
                                line.BeginValue = static_cast<float>(entryPrice);
                                line.AddMethod = UTAM_ADD_OR_ADJUST;
                                line.LineNumber = 99999;

                                sc.UseTool(line);
                            }
                        }
                    }
                }
            }
        }

        if (foundNewSignal && EnableAlertSound.GetYesNo()) {
            sc.PlaySound(1);
        }
    } else {
        // Start new HTTP request
        p_Data->requestID = sc.MakeHTTPRequest(url);
    }
}
