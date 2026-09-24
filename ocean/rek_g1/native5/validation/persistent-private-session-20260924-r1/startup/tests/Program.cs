using RekUiBridgeAgent;
var checks=0;
void Expect(bool value){checks++;if(!value)throw new Exception("startup contract check failed: "+checks);}
var intro=new IntroSkipFacts(true,true,true,true,true,true,false,true,true);
Expect(StartupMenuContract.IntroRejectReason(intro) is null);
foreach(var bad in new[]{intro with{IsolatedSpark=false},intro with{IdleControls=false},intro with{NoNetwork=false},
    intro with{AtIntro=false},intro with{ControllerAvailable=false},intro with{IntroActive=false},
    intro with{Finished=true},intro with{SkipShown=false},intro with{SkipEnabled=false}})
    Expect(StartupMenuContract.IntroRejectReason(bad) is not null);
var exit=new UnsupportedPairingExitFacts(true,true,true,true,true,true,true,true,true,true,true);
Expect(StartupMenuContract.UnsupportedExitRejectReason(exit) is null);
foreach(var bad in new[]{exit with{IsolatedSpark=false},exit with{IdleControls=false},exit with{PrivateBotOneNoHumanScope=false},
    exit with{GameMenuAvailable=false},exit with{MatchingSlots=false},exit with{VisualOnlyPair=false},
    exit with{LocalSemanticG1=false},exit with{OpponentSemanticT800=false},exit with{LocalExactG1Bones=false},
    exit with{OpponentExactT800Bones=false},exit with{MixedPairReason=false}})
    Expect(StartupMenuContract.UnsupportedExitRejectReason(bad) is not null);
var confirmation=new HomeForfeitConfirmationFacts(true,true,true,true,true);
Expect(StartupMenuContract.CanConfirmHomeForfeit(confirmation));
foreach(var bad in new[]{confirmation with{TargetHome=false},confirmation with{MenuOpen=false},
    confirmation with{AtForfeitPane=false},confirmation with{ButtonShown=false},confirmation with{ButtonEnabled=false}})
    Expect(!StartupMenuContract.CanConfirmHomeForfeit(bad));
Expect(!PolicyExecutionIsolationContract.WindowsCommandAllowed("SkipIntro"));
Expect(!PolicyExecutionIsolationContract.WindowsCommandAllowed("ExitUnsupportedPrivateAiPairing"));
Console.WriteLine(System.Text.Json.JsonSerializer.Serialize(new {passed=true,checks,game_connected=false,windows_input=false}));
