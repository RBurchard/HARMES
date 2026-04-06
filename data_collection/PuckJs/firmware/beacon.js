// This will send advertising packets every 50ms. No "beacon" implementation needed, just scan for ble devices with the correct manufacturer and manufacturerData.
NRF.setAdvertising({},{manufacturer: 0x0590, manufacturerData: 0x69}, {interval : 50});
