'use strict';

// Seats and mode rules, free of timers and native state.
// - One player: idle until they ask for Bot 1 (Bot 1 on fighter 1).
// - Two players: player versus player starts automatically.
// - A lone player always sits on fighter 0, the side Bot 1 fights against.
class Lobby{
  constructor(){this.seats=[null,null];this.spectators=new Set();this.wantsBot=false;}
  get players(){return this.seats.filter(Boolean).length;}
  seatOf(client){return this.seats.indexOf(client);}
  join(client){
    const free=this.seats.indexOf(null);
    if(free<0){this.spectators.add(client);return -1;}
    this.seats[free]=client;
    // A second player replaces any Bot 1 match.
    if(this.players===2)this.wantsBot=false;
    return free;
  }
  leave(client){
    this.spectators.delete(client);
    const seat=this.seatOf(client);if(seat<0)return false;
    this.seats[seat]=null;this.wantsBot=false;
    if(!this.seats[0]&&this.seats[1]){this.seats[0]=this.seats[1];this.seats[1]=null;}
    // A waiting spectator takes the free seat.
    const next=this.spectators.values().next();
    if(!next.done){this.spectators.delete(next.value);this.seats[this.seats.indexOf(null)]=next.value;}
    return true;
  }
  requestBot(client){
    if(this.players!==1||this.seatOf(client)!==0)return false;
    this.wantsBot=true;return true;
  }
  stopBot(){this.wantsBot=false;}
  mode(){
    if(this.players===2)return 'pvp';
    if(this.players===1&&this.wantsBot)return 'bot';
    return 'idle';
  }
}
module.exports={Lobby};
