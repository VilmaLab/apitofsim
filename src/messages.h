#pragma once

#include <sstream>
#include <iomanip>

template <typename Callback>
std::string call_with_stringstream(Callback cb)
{
  std::stringstream ss;
  ss << std::scientific << std::setprecision(3);
  cb(ss);
  return ss.str();
}

std::string prepare_message(const std::string &msg)
{
  return msg;
}

std::string prepare_message(const char *msg)
{
  return msg;
}

template <typename Callback>
std::string prepare_message(Callback cb)
{
  return call_with_stringstream(cb);
}
