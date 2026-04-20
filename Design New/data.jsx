// Capper sample data
const TEAMS = {
  ARS: { name: 'Arsenal', color: '#DC0714' },
  LIV: { name: 'Liverpool', color: '#C8102E' },
  MCI: { name: 'Man City', color: '#6CABDD' },
  CHE: { name: 'Chelsea', color: '#034694' },
  TOT: { name: 'Tottenham', color: '#132257' },
  NEW: { name: 'Newcastle', color: '#241F20' },
  AVL: { name: 'Aston Villa', color: '#670E36' },
  WHU: { name: 'West Ham', color: '#7A263A' },
  BRI: { name: 'Brighton', color: '#0057B8' },
  FUL: { name: 'Fulham', color: '#000000' },
  RMA: { name: 'Real Madrid', color: '#FEBE10' },
  BAR: { name: 'Barcelona', color: '#A50044' },
  ATM: { name: 'Atlético', color: '#CB3524' },
  BAY: { name: 'Bayern', color: '#DC052D' },
  BVB: { name: 'Dortmund', color: '#FDE100' },
  PSG: { name: 'PSG', color: '#004170' },
  INT: { name: 'Inter', color: '#0068A8' },
  JUV: { name: 'Juventus', color: '#000000' },
  MIL: { name: 'Milan', color: '#FB090B' },
  NAP: { name: 'Napoli', color: '#12A0D7' },
};

const TODAY_PICKS = [
  {
    id: 1, league: 'Premier League', time: '14:00', live: true, minute: "67'", score: '1-2',
    home: 'ARS', away: 'LIV', pick: 'AWAY', odds: 2.45, ev: 17, stake: 47, conf: 78,
  },
  {
    id: 2, league: 'La Liga', time: '17:30', live: false,
    home: 'RMA', away: 'BAR', pick: 'OVER 2.5', odds: 1.85, ev: 9, stake: 32, conf: 64,
  },
  {
    id: 3, league: 'Serie A', time: '19:45', live: false,
    home: 'INT', away: 'JUV', pick: 'HOME', odds: 1.92, ev: 38, stake: 38, conf: 71,
  },
  {
    id: 4, league: 'Bundesliga', time: '20:30', live: true, minute: "23'", score: '0-0',
    home: 'BAY', away: 'BVB', pick: 'BTTS', odds: 1.65, ev: 6, stake: 24, conf: 58,
  },
  {
    id: 5, league: 'Premier League', time: '21:15', live: false,
    home: 'MCI', away: 'CHE', pick: 'HOME -1', odds: 2.10, ev: 14, stake: 41, conf: 73,
  },
  {
    id: 6, league: 'Ligue 1', time: '22:00', live: false,
    home: 'PSG', away: 'MIL', pick: 'OVER 3.5', odds: 2.30, ev: 11, stake: 35, conf: 67,
  },
];

const SCHEDULE = [
  { date: '21 КВІ', dow: 'ПОНЕДІЛОК', count: 4, picks: [
    { league: 'Premier League', time: '20:00', home: 'NEW', away: 'TOT', pick: 'HOME', odds: 1.95, ev: 13 },
    { league: 'La Liga', time: '21:00', home: 'ATM', away: 'BAR', pick: 'UNDER 2.5', odds: 2.05, ev: 8 },
    { league: 'Serie A', time: '19:45', home: 'NAP', away: 'MIL', pick: 'BTTS', odds: 1.72, ev: 7 },
    { league: 'Bundesliga', time: '20:30', home: 'BVB', away: 'BAY', pick: 'OVER 2.5', odds: 1.55, ev: 5 },
  ]},
  { date: '22 КВІ', dow: 'ВІВТОРОК', count: 3, picks: [
    { league: 'Champions League', time: '21:00', home: 'RMA', away: 'MCI', pick: 'OVER 2.5', odds: 1.78, ev: 10 },
    { league: 'Champions League', time: '21:00', home: 'PSG', away: 'BAY', pick: 'AWAY', odds: 2.55, ev: 16 },
    { league: 'Premier League', time: '19:30', home: 'AVL', away: 'WHU', pick: 'HOME', odds: 1.65, ev: 6 },
  ]},
  { date: '23 КВІ', dow: 'СЕРЕДА', count: 2, picks: [
    { league: 'Champions League', time: '21:00', home: 'BAR', away: 'INT', pick: 'BTTS', odds: 1.60, ev: 9 },
    { league: 'Europa League', time: '19:00', home: 'CHE', away: 'JUV', pick: 'HOME', odds: 2.20, ev: 12 },
  ]},
  { date: '24 КВІ', dow: 'ЧЕТВЕР', count: 5, picks: [
    { league: 'Europa League', time: '21:00', home: 'TOT', away: 'MIL', pick: 'AWAY', odds: 2.85, ev: 19 },
    { league: 'Conference', time: '21:00', home: 'BRI', away: 'FUL', pick: 'OVER 2.5', odds: 1.90, ev: 11 },
    { league: 'Eredivisie', time: '20:00', home: 'ARS', away: 'LIV', pick: 'HOME', odds: 1.88, ev: 8 },
    { league: 'Primeira', time: '21:30', home: 'PSG', away: 'NAP', pick: 'OVER 3.5', odds: 2.40, ev: 14 },
    { league: 'Pro League', time: '20:30', home: 'NEW', away: 'AVL', pick: 'BTTS', odds: 1.75, ev: 7 },
  ]},
];

// Sparkline data — bankroll growth over 30 days
const SPARK_DATA = [
  1000, 1015, 1003, 1042, 1028, 1071, 1085, 1063, 1097, 1112,
  1098, 1124, 1141, 1135, 1158, 1172, 1156, 1183, 1198, 1175,
  1209, 1221, 1208, 1235, 1247, 1228, 1251, 1239, 1238, 1243,
];

// Streak — last 15 bets
const STREAK = ['W','W','L','W','W','W','P','W','L','W','W','W','L','W','W'];

// Curve data — 90 days of profit (pct)
const CURVE_DATA = [
  0, 1.2, -0.5, 2.1, 1.8, 3.4, 4.1, 3.6, 5.2, 6.8,
  6.1, 7.5, 8.3, 7.9, 9.1, 10.4, 9.7, 11.2, 12.5, 11.8,
  13.4, 14.7, 13.9, 15.2, 16.8, 15.5, 17.3, 16.4, 18.1, 19.6,
  18.7, 20.3, 21.5, 20.8, 22.4, 23.7, 22.9, 24.3,
];

window.CAPPER_DATA = { TEAMS, TODAY_PICKS, SCHEDULE, SPARK_DATA, STREAK, CURVE_DATA };
